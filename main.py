from training import Training
from evaluation import Evaluation
from image_sampling import ImageSampler
import os
from torchvision.utils import save_image


def main():
    training = Training()
    print(training.cpu_available())
    #training.train()
    #evaluation = Evaluation()
    #evaluation.evaluate()

    '''
    sampler = ImageSampler("checkpoints/flow_mnist.pt")
    imgs1 = sampler.sample(64)
    imgs2 = sampler.sample(32, temperature=0.7)  # Schärfere Bilder
    imgs3 = sampler.sample(100, temperature=1.2)  # Diversere Bilder
    os.makedirs("outputs", exist_ok=True)
    save_image(imgs1, "outputs/samples_default.png", nrow=8)
    save_image(imgs2, "outputs/samples_temp07.png", nrow=8)
    save_image(imgs3, "outputs/samples_temp12.png", nrow=10)
    '''

if __name__ == "__main__":
    main()
