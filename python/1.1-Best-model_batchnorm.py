classifier_input = Input(shape=input_shape)

x = Conv2D(32, (3, 3), activation='relu', padding='same')(classifier_input)
x = MaxPooling2D()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Flatten()(x)
x = Dense(num_classes, activation='softmax')(x)

model_batchnorm = Model(outputs=x, inputs=classifier_input)
model_batchnorm.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model_batchnorm.summary()

lr = 0.001
K.set_value(model_batchnorm.optimizer.lr, lr)
model_batchnorm.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=10,
        validation_data=validation_generator,
        validation_steps=60)

lr = 0.001
K.set_value(model_batchnorm.optimizer.lr, lr)
model_batchnorm.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=60)

lr = 0.001
K.set_value(model_batchnorm.optimizer.lr, lr)
model_batchnorm.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=60)

lr = 0.0001
K.set_value(model_batchnorm.optimizer.lr, lr)
model_batchnorm.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=2,
        validation_data=validation_generator,
        validation_steps=60)





