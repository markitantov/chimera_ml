import sys

from chimera_ml.cli import app as chimera_main
import chimera_plugin


def main() -> None:
    chimera_plugin.register()
    chimera_main(sys.argv[1:])


if __name__ == "__main__":
    main()
