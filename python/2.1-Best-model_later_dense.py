x = Conv2D(32, (3, 3), activation='relu', padding='same')(classifier_input)
x = MaxPooling2D()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = BatchNormalization(axis=-1)(x)
x = MaxPooling2D()(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = Flatten()(x)
x = merge([x, dist2land_input], 'concat')
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
out = Dense(num_classes, activation='softmax')(x)

model_dense = Model(inputs=[classifier_input, dist2land_input], outputs=out)
model_dense.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model_dense.summary()

SVG(model_to_dot(model_dense).create(prog='dot', format='svg'))

lr = 0.001
K.set_value(model_dense.optimizer.lr, lr)
model_dense.fit([np_train_crops, np_train_feature], np_train_class,
          batch_size=32,
          epochs=10,
          validation_data=([np_valid_crops, np_valid_feature], np_valid_class))

lr = 0.001
K.set_value(model_dense.optimizer.lr, lr)
model_dense.fit([np_train_crops, np_train_feature], np_train_class,
          batch_size=32,
          epochs=5,
          validation_data=([np_valid_crops, np_valid_feature], np_valid_class))

lr = 0.0001
K.set_value(model_dense.optimizer.lr, lr)
model_dense.fit([np_train_crops, np_train_feature], np_train_class,
          batch_size=32,
          epochs=5,
          validation_data=([np_valid_crops, np_valid_feature], np_valid_class))

lr = 0.00001
K.set_value(model_dense.optimizer.lr, lr)
model_dense.fit([np_train_crops, np_train_feature], np_train_class,
          batch_size=32,
          epochs=2,
          validation_data=([np_valid_crops, np_valid_feature], np_valid_class))

