from matplotlib import pyplot as plt
from sconce.data_generator import DataGenerator
from sconce.models import BasicClassifier
from sconce.trainers import ClassifierTrainer
from torch import optim
from torchvision import datasets

import sconce
import torch

print(f'Sconce Version: {sconce.__version__}')

def get_trainer(dataset_class=datasets.MNIST, batch_size=500):
    model = BasicClassifier.new_from_yaml_filename('basic_classifier_MNIST.yaml')

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

survey_monitor = trainer.survey_learning_rate(
    num_epochs=1, min_learning_rate=1e-4, max_learning_rate=10, stop_factor=3)
survey_monitor.dataframe_monitor.plot_learning_rate_survey(figure_kwargs={'figsize': (5, 3)});

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(1, 1, 1)
for i in range(20):
    survey_monitor = trainer.survey_learning_rate(
        num_epochs=1, min_learning_rate=1e-4, max_learning_rate=10, stop_factor=3)
    survey_monitor.dataframe_monitor.plot_learning_rate_survey(ax=ax, color='#1f77b4', alpha=0.8);

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(1, 1, 1)
for i in range(20):
    trainer = get_trainer()
    survey_monitor = trainer.survey_learning_rate(
        num_epochs=1, min_learning_rate=1e-4, max_learning_rate=10, stop_factor=3)
    survey_monitor.dataframe_monitor.plot_learning_rate_survey(ax=ax, color='#1f77b4', alpha=0.8);

fig = plt.figure(figsize=(6,4))
ax = fig.add_subplot(1, 1, 1)
for i in range(10):
    trainer = get_trainer()
    survey_monitor = trainer.survey_learning_rate(
        num_epochs=1, min_learning_rate=1e-4, max_learning_rate=10, stop_factor=3)
    survey_monitor.dataframe_monitor.plot_learning_rate_survey(ax=ax,
        label='MNIST', color='#1f77b4', alpha=0.8);
for i in range(10):
    trainer = get_trainer(dataset_class=datasets.FashionMNIST)
    survey_monitor = trainer.survey_learning_rate(
        num_epochs=1, min_learning_rate=1e-4, max_learning_rate=10, stop_factor=3)
    survey_monitor.dataframe_monitor.plot_learning_rate_survey(ax=ax,
        label='FashionMNIST', color='#ff7f0e', alpha=0.8)
ax.legend();

trainer = get_trainer()
fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(1, 1, 1)
for i in range(1, 6):
    survey_monitor = trainer.survey_learning_rate(
        num_epochs=i, min_learning_rate=1e-4, max_learning_rate=10, stop_factor=3)
    survey_monitor.dataframe_monitor.plot_learning_rate_survey(ax=ax, label=f'Epochs: {i}')
ax.legend();

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(1, 1, 1)
for bs in [25, 50, 100, 200, 500, 1000]:
    trainer = get_trainer(batch_size=bs)
    
    survey_monitor = trainer.survey_learning_rate(
        num_epochs=1, min_learning_rate=1e-4, max_learning_rate=10, stop_factor=3)
    survey_monitor.dataframe_monitor.plot_learning_rate_survey(ax=ax, label=f'Batch Size: {bs}')
ax.legend();

fig = plt.figure(figsize=(7, 5))
ax = fig.add_subplot(1, 1, 1)
for bs in [25, 50, 100, 200, 500, 1000]:
    trainer = get_trainer(batch_size=bs)

    num_epochs = bs / 200
    survey_monitor = trainer.survey_learning_rate(
        num_epochs=num_epochs, min_learning_rate=1e-4, max_learning_rate=10, stop_factor=3)
    survey_monitor.dataframe_monitor.plot_learning_rate_survey(ax=ax, label=f'Batch Size: {bs}')
ax.legend();



