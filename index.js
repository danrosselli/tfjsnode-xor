
import tf from '@tensorflow/tfjs-node';
import termkit from 'terminal-kit';

const term = termkit.terminal;

term.clear();

// Definir os dados de treinamento
const trainingData = tf.tensor2d([
  [0, 0],
  [0, 1],
  [1, 0],
  [1, 1]
]);

const targets = tf.tensor2d([
  [0],
  [1],
  [1],
  [0]
]);

// Definir o modelo
const model = tf.sequential();
model.add(tf.layers.dense({ units: 8, inputShape: [2], activation: 'sigmoid' }));
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

// Compilar o modelo
model.compile({
  optimizer: tf.train.adam(0.02),
  loss: 'meanSquaredError'
});

// Adicione um callback onEpochEnd
const onEpochEndCallback = {
  onEpochEnd: async (epoch, logs) => {
    console.log(`Epoch ${epoch + 1}, Loss: ${logs.loss}`);
    term.previousLine(1);
  }
};

// Treinar o modelo
model.fit(trainingData, targets, { epochs: 5000, verbose: 0, callbacks: onEpochEndCallback }).then(() => {
  
  // Teminou o treino, então agora é testar o modelo treinado
  const testInput = tf.tensor2d([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
  ]);

  term.nextLine(1);
  const predictions = model.predict(testInput).arraySync();
  
  // Iterar sobre as previsões
  console.log('--------------------------------------');
  predictions.forEach((prediction, index) => {
    console.log(`Input: ${testInput.arraySync()[index]}, Prediction: ${prediction}`);
  });

});
