from pathlib import Path


class Maestro:
    models_path = Path(__file__).parent / 'models'
    configs_path = Path(__file__).parent / 'configs'
    output_path = Path(__file__).parent / 'output'