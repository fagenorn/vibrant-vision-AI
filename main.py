from vibrant_vision.sd.models.Model import Model

def main():
    model = Model()
    result = model.predict("a photo of an astronaut riding a horse on mars")

    for i, image in enumerate(result.images):
        image.save(f"image_{i}.png")

if __name__ == "__main__":
    main()
    