from vibrant_vision.sd.Maestro import Maestro
from vibrant_vision.sd.models.ImageModel import ImageModel
from vibrant_vision.general.logging import setup_logging


def main():
    maestro = Maestro(fps=30)
    maestro.perform()
    # model = Model()
    # result = model.predict("a photo of an astronaut riding a horse on mars")

    # for i, image in enumerate(result.images):
    #     image.save(f"image_{i}.png")


if __name__ == "__main__":
    setup_logging()
    main()
