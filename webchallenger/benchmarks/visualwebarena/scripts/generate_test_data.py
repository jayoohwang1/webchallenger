"""Replace the website placeholders with website domains from env_config
Generate the test data"""
import json
import os

#from browser_env.env_config import *


def main() -> None:
    inp_paths = [
        "config_files/test_classifieds.raw.json", "config_files/test_shopping.raw.json", "config_files/test_reddit.raw.json",
        "config_files/test_webarena.raw.json"  # Uncomment to generate test files for WebArena
        # "config_files/test_webarena_lite.raw.json"
    ]

    for inp_path in inp_paths:
        inp_path = f'webchallenger/benchmarks/visualwebarena/{inp_path}'
        output_dir = inp_path.replace('.raw.json', '')
        os.makedirs(output_dir, exist_ok=True)
        with open(inp_path, "r") as f:
            raw = f.read()
        raw = raw.replace("__REDDIT__", os.environ.get('REDDIT'))
        raw = raw.replace("__SHOPPING__", os.environ.get('SHOPPING'))
        raw = raw.replace("__WIKIPEDIA__", os.environ.get('WIKIPEDIA'))
        raw = raw.replace("__CLASSIFIEDS__", os.environ.get('CLASSIFIEDS'))

        if ("test_webarena.raw.json" in inp_path) or ("test_webarena_lite.raw.json" in inp_path):
            raw = raw.replace("__GITLAB__", os.environ.get('GITLAB'))
            raw = raw.replace("__SHOPPING_ADMIN__", os.environ.get('SHOPPING_ADMIN'))
            raw = raw.replace("__MAP__", os.environ.get('MAP'))

        with open(inp_path.replace(".raw", ""), "w") as f:
            f.write(raw)
        data = json.loads(raw)
        for idx, item in enumerate(data):
            with open(os.path.join(output_dir, f"{idx}.json"), "w") as f:
                json.dump(item, f, indent=2)


if __name__ == "__main__":
    main()
