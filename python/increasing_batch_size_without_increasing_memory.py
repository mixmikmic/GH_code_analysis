from matplotlib import pyplot as plt
from sconce.data_generator import DataGenerator
from sconce.models import MultilayerPerceptron
from sconce.trainers import ClassifierTrainer
from torch import optim
from torchvision import datasets

import sconce
import torch

print(f'Sconce Version: {sconce.__version__}')

def get_trainer(dataset_class=datasets.MNIST, batch_size=100):
    model = MultilayerPerceptron.new_from_yaml_filename('multilayer_perceptron_MNIST.yaml')

    training_generator = DataGenerator.from_pytorch(batch_size=batch_size,
                                                    dataset_class=dataset_class)
    test_generator = DataGenerator.from_pytorch(batch_size=batch_size,
                                                dataset_class=dataset_class,
                                                train=False)

    if torch.cuda.is_available():
        model.cuda()
        training_generator.cuda()
        test_generator.cuda()

    optimizer = optim.SGD(model.parameters(), lr=1e-2,
            momentum=0.9, weight_decay=1e-4)

    trainer = ClassifierTrainer(model=model, optimizer=optimizer,
        training_data_generator=training_generator,
        test_data_generator=test_generator)
    return trainer

trainer = get_trainer()

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(1, 1, 1)

# real large batches
for bs in [100, 1_000, 10_000]:
    trainer = get_trainer(batch_size=bs)

    num_epochs = bs / 200
    survey_monitor = trainer.survey_learning_rate(
        num_epochs=num_epochs, min_learning_rate=1e-4, max_learning_rate=10, stop_factor=3)
    survey_monitor.dataframe_monitor.plot_learning_rate_survey(ax=ax, label=f'Real Batch Size: {bs}')

# virtual large batches
for bm in [1, 10, 100]:
    bs = 100
    trainer = get_trainer(batch_size=bs)

    ebs = bs * bm
    num_epochs = ebs / 200
    survey_monitor = trainer.survey_learning_rate(
        num_epochs=num_epochs, min_learning_rate=1e-4, max_learning_rate=10, stop_factor=3, batch_multiplier=bm)
    survey_monitor.dataframe_monitor.plot_learning_rate_survey(ax=ax, label=f'Effective Batch Size: {ebs}')
ax.legend();



