"""UNSW-NB15 preparation entry point."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from preprocess.prepare import main

if __name__ == '__main__':
    main(['--dataset', 'UNSW-NB15', *sys.argv[1:]])
