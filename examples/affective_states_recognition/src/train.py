import sys

import chimera_plugin

from chimera_ml.cli import app as chimera_main


def main() -> None:
    chimera_plugin.register()
    chimera_main(sys.argv[1:])


if __name__ == "__main__":
    main()
