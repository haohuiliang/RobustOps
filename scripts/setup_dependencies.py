"""Prepare upstream folders at the exact commits recorded in .gitmodules."""
import argparse
import configparser
import json
from pathlib import Path
import subprocess


def setup(root, dry_run=False):
    config = configparser.ConfigParser()
    config.read(root / '.gitmodules')
    for section in config.sections():
        item = config[section]
        destination = root / item['path']
        revision = item['revision']
        def run(command):
            print(json.dumps(command), flush=True)
            if not dry_run:
                subprocess.run(command, check=True)
        if not (destination / '.git').exists():
            run(['git', 'clone', '--no-checkout', item['url'], str(destination)])
            run(['git', '-C', str(destination), 'checkout', '--detach', revision])
        else:
            head = subprocess.check_output(['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
            if head != revision:
                raise ValueError(f'{destination} is at {head}; expected {revision}. Preserve local work and use a fresh checkout.')
        if item['path'] == 'verl':
            patch = str(root / 'scripts/verl.patch')
            if destination.exists():
                applied = subprocess.run(['git', '-C', str(destination), 'apply', '--reverse', '--check', patch], capture_output=True).returncode == 0
                if applied:
                    print('veRL patch already applied')
                    continue
                clean = subprocess.run(['git', '-C', str(destination), 'apply', '--check', patch], capture_output=True).returncode == 0
                if not clean:
                    # Existing installations may have the previous project patch.
                    # Apply only its incremental update, preserving other local work.
                    upgrade = str(root / 'scripts/verl-upgrade.patch')
                    compatible = subprocess.run(['git', '-C', str(destination), 'apply', '--check', upgrade], capture_output=True).returncode == 0
                    if not compatible:
                        raise ValueError(f'{destination} has conflicting local changes. Preserve them and use a fresh checkout before applying the veRL patch.')
                    patch = upgrade
            run(['git', '-C', str(destination), 'apply', '--check', patch])
            run(['git', '-C', str(destination), 'apply', patch])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    setup(Path(__file__).resolve().parents[1], args.dry_run)
