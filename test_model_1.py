import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
import json as js


from TrainDataGenerator import train_data_generator
from InputOutputGenerator import InputOutputGenerator
from DataContainer import FileDataContainer


#input_dim = (stations_number*2 + 2) * num_segments

with open("data.txt", 'r') as f:
    data = js.load(f)

def init_generators(data):
    random_data_generator = InputOutputGenerator()
    random_data_generator.num_segments = data['num_segments']
    random_data_generator.stations_number = data['stations_number']
    random_data_generator.output_dim = data['output_dim']
    random_data_generator.train_density = data['train_density']
    random_data_generator.set_matrix()
    
    container = FileDataContainer(data['batch_size'] * data['steps_per_epoch'])
    container.inputs_dir = data['inputs_dir']
    container.outputs_dir = data['outputs_dir']
    
    val_container = FileDataContainer(data['validation_batch_size'] * data['validation_steps'])
    val_container.inputs_dir = data['val_inputs_dir']
    val_container.outputs_dir = data['val_outputs_dir']
    
    if data['files_exist']:
        container.is_initialized = True #  container.create_data_set(random_data_generator)
        container.is_initialized = True
    else:
        container.create_data_set(random_data_generator)
        val_container.create_data_set(random_data_generator)
    generator = train_data_generator(data['batch_size'], container)
    val_generator = train_data_generator(data['validation_batch_size'], val_container)
    return (generator, val_generator, *random_data_generator.get_dimensions())

train_generator, val_generator, input_dim, output_dim = init_generators(data)

model = Sequential()
model.add(Dense(data['neurons_num'], activation='relu', input_shape=(input_dim,)))
model.add(Dense(output_dim))

model.compile(loss='mean_absolute_error', metrics=['mean_absolute_percentage_error'])

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=data['steps_per_epoch'],
                              epochs=data['epochs'],
                              validation_data=val_generator,
                              validation_steps=data['validation_steps'])
        
history_dict = history.history
with open("result.txt", "w") as f:
    js.dump(history_dict, f)

loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, data['epochs']+1)    

plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

metrics_values = history_dict['mean_absolute_percentage_error']
val_metrics_values = history_dict['val_mean_absolute_percentage_error']

epochs = range(1, data['epochs']+1)    

plt.plot(epochs, metrics_values, 'bo', label='Training loss')
plt.plot(epochs, val_metrics_values, 'b', label='Validation loss')
plt.title('Training and validation metrics')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

