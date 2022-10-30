from glob import glob

HTML_CONTENT = """
<div class="results">
    <video autoplay loop muted playsinline width="100%">
        <source src="./resources/videos/{video_path}" type="video/mp4">
    </video>
</div>
"""


def generate_html(video_path):
    return HTML_CONTENT.format(video_path=video_path)


if __name__ == "__main__":
    for path in glob("*.mp4"):
        print(generate_html(path))
