import numpy as np
import torch
from tqdm.auto import tqdm

from relpose.utils.geometry import generate_random_rotations


def get_permutations(num_frames):
    permutations = []
    for i in range(num_frames):
        for j in range(num_frames):
            if i != j:
                permutations.append((i, j))
    return torch.tensor(permutations)


def score_hypothesis(hypothesis, model, permutations, features):
    R_pred_batched = hypothesis[permutations]
    R_pred_rel = torch.einsum(
        "bij,bjk ->bik",
        R_pred_batched[:, 0].permute(0, 2, 1),
        R_pred_batched[:, 1],
    )
    features_batched = features[permutations]
    _, logits = model(
        features1=features_batched[:, 0],
        features2=features_batched[:, 1],
        queries=R_pred_rel.unsqueeze(0),
    )
    score = torch.trace(logits)
    return score


def compute_mst(num_frames, best_probs, best_rotations):
    """
    Computes the maximum spanning tree of the graph defined by the best_probs. Uses
    Prim's algorithm (modified for a directed graph).
    Currently a naive O(N^3) implementation :P
    """
    current_assigned = {0}
    assigned_rotations = np.tile(np.eye(3), [num_frames, 1, 1])

    edges = []

    while len(current_assigned) < num_frames:
        # Find the highest probability edge that connects an unassigned node to the MST
        best_i = -1
        best_j = -1
        best_p = -1
        not_assigned = set(range(num_frames)) - current_assigned
        for i in current_assigned:
            for j in not_assigned:
                if best_probs[i, j] > best_p:
                    best_p = best_probs[i, j]
                    best_i = i
                    best_j = j
                if best_probs[j, i] > best_p:
                    best_p = best_probs[j, i]
                    best_i = j
                    best_j = i

        rot = best_rotations[best_i, best_j]
        if best_i in current_assigned:
            current_assigned.add(best_j)
            assigned_rotations[best_j] = assigned_rotations[best_i] @ rot
        else:
            current_assigned.add(best_i)
            assigned_rotations[best_i] = assigned_rotations[best_j] @ rot.T
        edges.append((best_i, best_j))

    return assigned_rotations, edges


def run_maximum_spanning_tree(model, images, num_frames):
    device = images.device
    permutations = get_permutations(num_frames)
    best_rotations = np.zeros((num_frames, num_frames, 3, 3))
    best_probs = np.zeros((num_frames, num_frames))
    for i, j in permutations:
        image1 = images[i].unsqueeze(0).to(device)
        image2 = images[j].unsqueeze(0).to(device)
        with torch.no_grad():
            queries, logits = model(
                images1=image1,
                images2=image2,
            )
        probabilities = torch.softmax(logits, -1)
        probabilities = probabilities[0].detach().cpu().numpy()
        best_prob = probabilities.max()
        best_rotation = queries[0].detach().cpu().numpy()[probabilities.argmax()]

        best_rotations[i, j] = best_rotation
        best_probs[i, j] = best_prob

    rotations_pred, edges = compute_mst(
        num_frames=num_frames,
        best_probs=best_probs,
        best_rotations=best_rotations,
    )
    return rotations_pred


def run_coordinate_ascent(
    model,
    images,
    num_frames,
    initial_hypothesis,
    num_iterations=200,
    num_queries=250_000,
    use_pbar=True,
):
    """
    Args:
        model (nn.Module): RelPose model.
        images (torch.Tensor): Tensor of shape (N, 3, H, W) containing the images.
        num_frames (int): Number of frames in the sequence.
        initial_hypothesis (np.ndarray): Initial hypothesis of shape (N, 3, 3).
        num_iterations (int): Number of iterations to run coordinate ascent. Defaults
            to 200.
        num_queries (int): Number of queries to use for each coordinate ascent. Defaults
            to 250,000.
        use_pbar (bool): Whether to use a progress bar. Defaults to True.

    Returns:
        torch.tensor: Final hypothesis of shape (N, 3, 3).
    """
    device = images.device
    hypothesis = torch.from_numpy(initial_hypothesis).to(device).float()
    features = model.feature_extractor(images.to(device))
    it = tqdm(range(num_iterations)) if use_pbar else range(num_iterations)
    for j in it:
        # Randomly sample an index to update
        k = np.random.choice(num_frames)
        proposals = generate_random_rotations(num_queries, device)
        proposals[0] = hypothesis[k]
        scores = torch.zeros(1, num_queries, device=device)
        for i in range(num_frames):
            if i == k:
                continue
            feature1 = features[i, None]
            feature2 = features[k, None]
            R_rel = hypothesis[i].T @ proposals
            with torch.no_grad():
                _, logits = model(
                    features1=feature1,
                    features2=feature2,
                    queries=R_rel.unsqueeze(0),
                )
                scores += logits
                _, logits = model(
                    features1=feature2,
                    features2=feature1,
                    queries=R_rel.transpose(1, 2).unsqueeze(0),
                )
                scores += logits
        best_ind = scores.argmax()
        hypothesis[k] = proposals[best_ind]
    return hypothesis
