import fastchat.serve.cli
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=str,
        default="lmsys/fastchat-t5-3b-v1.0",
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank.")
    # parser.add_argument("--deepspeed", action="store_true", help="Use deepspeed.")
    parser.add_argument("--deepspeed", default=True, help="Use deepspeed.")

    parser.add_argument("--offload", action="store_true", help="Use deepspeed zero-offloading.")

    parser.add_argument("--meta", action="store_true", help="Use deepspeed meta.")
    args = parser.parse_args()
    fastchat.serve.cli.main(args)