import argparse

from types import SimpleNamespace

from webchallenger.agent import add_agent_args, run_agent


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--explore_websites", help="Explore all websites for a given benchmark or provide url list")
    parser.add_argument("--benchmark", help="Benchmark to evaluate", default=None,
                        choices=["webarena", "visualwebarena", "workarena_l1", "online_mind2web"])
    parser.add_argument("--intent", help="Task instruction for agent")
    
    parser.add_argument("--vision",
                        help="Vision model option", choices=["cogagent", "qwen2_vl", "molmo_7b", "internvl_8b", "os_atlas_7b", "qwen2.5vl", "holo1"], default="qwen2.5vl")
    parser.add_argument(
        "--vlm_dir", help="Directory of vlm model weights", default=None)
    parser.add_argument("--planning", help="Planning model option",
                        choices=["mistral7b", "llama3_8b", "mistral_small", "solar_pro_preview", "qwen_14b", "qwen3", "glm4", "gemini"], default="glm4")
    parser.add_argument(
        "--llm_dir", help="Directory of llm model weights", default=None)
    parser.add_argument("--browser", help="Browser option",
                        choices=["firefox", "chromium", "webkit"], default="chromium")
    parser.add_argument("--headless", action="store_true",
                        help="Run in headless mode")
    parser.add_argument("--vlm-temp", default=0.0,
                        help="Temperature of vlm")
    parser.add_argument("--llm-temp", default=0.0,
                        help="Temperature of llm")
    parser.add_argument("--do_sample", action="store_true", default=False,
                        help="Whether to perform sampling during model generation")

    
def add_server_args(parser: argparse.ArgumentParser):
    parser.add_argument("--local_backup", default=True,
                        help="revert to local model if LLM API call fails")
    parser.add_argument("--llm_api_url", default="http://localhost:5000",
                        help="URL for OpenAI-compatible LLM API endpoint (default = http://localhost:5000)")
    parser.add_argument("--llm_api_model", default=None,
                        help="LLM 'model' name argument for external API call (default=None)")
    
    parser.add_argument("--vlm_api_url", default="http://localhost:5001",
                        help="URL for OpenAI-compatible VLM API endpoint (default = http://localhost:5001)")
    parser.add_argument("--vlm_api_model", default=None,
                        help="VLM 'model' name argument for external API call (default=None)")


def parse_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Webstudent command-line interface")

    add_common_args(parser)
    add_agent_args(parser)
    add_server_args(parser)

    return SimpleNamespace(**parser.parse_args().__dict__)


def main():
    args = parse_arg()
    run_agent(args.vision, args.planning, args.browser, args)


if __name__ == "__main__":
    main()
