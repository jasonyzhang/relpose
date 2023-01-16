import antialiased_cnns
import numpy as np
import torch
import torch.nn as nn

from relpose.utils.geometry import generate_random_rotations, generate_superfibonacci


def generate_hypotheses(rotations_gt, num_queries=50000):
    """
    Args:
        rotations_gt (tensor): Batched rotations (B, N_I, 3, 3).

    Returns:
        hypotheses (tensor): Hypotheses (B, N_I, N_Q, 3, 3).
    """
    batch_size, num_images, _, _ = rotations_gt.shape
    hypotheses = generate_random_rotations(
        (num_queries - 1) * batch_size * num_images, device=rotations_gt.device
    )
    # (B, N_i, N_q - 1, 3, 3)
    hypotheses = hypotheses.reshape(batch_size, num_images, (num_queries - 1), 3, 3)
    # (B, N_i, N_q, 3, 3)
    hypotheses = torch.cat((rotations_gt.unsqueeze(2), hypotheses), dim=2)
    return hypotheses


def get_feature_extractor():
    """
    Returns a network that takes in images (B, 3, 224, 224) and outputs a feature
    vector (B, 2048, 1, 1).
    """
    model = antialiased_cnns.resnet50(pretrained=True)
    feature_extractor = torch.nn.Sequential(*(list(model.children())[:-1]))
    return feature_extractor


class RelPose(nn.Module):
    def __init__(
        self,
        feature_extractor=None,
        num_pe_bases=8,
        num_layers=4,
        hidden_size=256,
        num_queries=50000,
        sample_mode="random",
        freeze_encoder=False,
    ):
        """
        Args:
            feature_extractor (nn.Module): Feature extractor.
            num_pe_bases (int): Number of positional encoding bases.
            num_layers (int): Number of layers in the network.
            hidden_size (int): Size of the hidden layer.
            num_queries (int): Number of rotations to sample if using random sampling.
            sample_mode (str): Sampling mode. Can be equivolumetric or random.
        """
        super().__init__()
        if feature_extractor is None:
            feature_extractor = get_feature_extractor()
        self.num_queries = num_queries
        self.sample_mode = sample_mode

        self.feature_extractor = feature_extractor
        if freeze_encoder:
            self.freeze_encoder()

        self.use_positional_encoding = num_pe_bases > 0
        if self.use_positional_encoding:
            query_size = num_pe_bases * 2 * 9
            self.register_buffer(
                "embedding", (2 ** torch.arange(num_pe_bases)).reshape(1, 1, -1)
            )
        else:
            query_size = 9

        self.embed_feature = nn.Linear(2048 * 2, hidden_size)
        self.embed_query = nn.Linear(query_size, hidden_size)
        layers = []
        for _ in range(num_layers - 2):
            layers.append(nn.LeakyReLU())
            layers.append(nn.Linear(hidden_size, hidden_size))
        # Yes, I forgot to add a non-linearity here. This will be fixed later.
        layers.append(nn.Linear(hidden_size, 1))
        self.layers = nn.Sequential(*layers)
        self.equi_grid = {}

    def freeze_encoder(self):
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

    def positional_encoding(self, x):
        """
        Args:
            x (tensor): Input (B, D).

        Returns:
            y (tensor): Positional encoding (B, 2 * D * L).
        """
        if not self.use_positional_encoding:
            return x
        embed = (x[..., None] * self.embedding).view(*x.shape[:-1], -1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)

    def forward(
        self,
        images1=None,
        images2=None,
        features1=None,
        features2=None,
        gt_rotation=None,
        num_queries=None,
        queries=None,
    ):
        """
        Must provide either images1 and images2 or features1 and features2. If
        gt_rotation is provided, the first query will be the ground truth rotation.

        Args:
            images1 (tensor): First set of images (B, 3, 224, 224).
            images2 (tensor): Corresponding set of images (B, 3, 224, 224).
            gt_rotation (tensor): Ground truth rotation (B, 3, 3).
            num_queries (int): Number of rotations to sample if using random sampling.

        Returns:
            rotations (tensor): Rotation matrices (B, num_queries, 3, 3). First query
                is the ground truth rotation.
            logits (tensor): logits (B, num_queries).
        """

        if features1 is None:
            features1 = self.feature_extractor(images1)
        if features2 is None:
            features2 = self.feature_extractor(images2)
        features = torch.cat([features1, features2], dim=1)

        batch_size = features1.size(0)
        assert batch_size == features2.size(0)
        features = features.reshape(batch_size, -1)  # (B, 4096)
        if queries is None:
            if num_queries is None:
                num_queries = self.num_queries
            if self.sample_mode == "equivolumetric":
                if num_queries not in self.equi_grid:
                    self.equi_grid[num_queries] = generate_superfibonacci(
                        num_queries, device="cpu"
                    )
                queries = self.equi_grid[num_queries].to(images1.device)
            elif self.sample_mode == "random":
                queries = generate_random_rotations(num_queries, device=images1.device)
            else:
                raise Exception(f"Unknown sampling mode {self.sample_mode}.")

            if gt_rotation is not None:
                delta_rot = queries[0].T @ gt_rotation
                # First entry will always be the gt rotation
                queries = torch.einsum("aij,bjk->baik", queries, delta_rot)
            else:
                if len(queries.shape) == 3:
                    queries = queries.unsqueeze(0)
                num_queries = queries.shape[1]
        else:
            num_queries = queries.shape[1]

        queries_pe = self.positional_encoding(queries.reshape(-1, num_queries, 9))

        e_f = self.embed_feature(features).unsqueeze(1)  # (B, 1, H)
        e_q = self.embed_query(queries_pe)  # (B, n_q, H)
        out = self.layers(e_f + e_q)  # (B, n_q, 1)
        logits = out.reshape(batch_size, num_queries)
        return queries, logits

    def predict_probability(
        self, images1, images2, query_rotation, recursion_level=None, num_queries=None
    ):
        """
        Args:
            images1 (tensor): First set of images (B, 3, 224, 224).
            images2 (tensor): Corresponding set of images (B, 3, 224, 224).
            gt_rotation (tensor): Ground truth rotation (B, 3, 3).
            num_queries (int): Number of rotations to sample. If gt_rotation is given
                will sample num_queries - batch size.

        Returns:
            probabilities
        """
        logits = self.forward(
            images1,
            images2,
            gt_rotation=query_rotation,
            num_queries=num_queries,
            recursion_level=recursion_level,
        )
        probabilities = torch.softmax(logits, dim=-1)
        probabilities = probabilities * num_queries / np.pi**2
        return probabilities[:, 0]
