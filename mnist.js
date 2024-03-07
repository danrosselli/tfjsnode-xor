import tf from '@tensorflow/tfjs-node';
import termkit from 'terminal-kit';
import jpeg from 'jpeg-js';
import fs from 'fs';
import mnist from 'mnist';

const term = termkit.terminal;
term.clear();


const set = mnist.set(60000, 0);

const convertToTensor = (data) => {
  const values = new Float32Array(data.flat());
  return tf.tensor3d(values, [data.length, 28, 28]);
};

const flatten = (arr) => arr.reduce((acc, val) => acc.concat(val), []);

const trainingData = convertToTensor(set.training.map(data => data.input));
const trainingLabels = tf.tensor1d(flatten(set.training.map(data => data.output)), 'int32');

const testData = convertToTensor(set.test.map(data => data.input));
const testLabels = tf.tensor1d(flatten(set.test.map(data => data.output)), 'int32');

console.log('Shape do tensor de treino:', trainingData.shape);
console.log('Shape do tensor de rótulos de treino:', trainingLabels.shape);
console.log('Shape do tensor de teste:', testData.shape);
console.log('Shape do tensor de rótulos de teste:', testLabels.shape);


/*
// load image 28x28x4(RGBA)
const imagePath = './dataset/testSample/img_1.jpg';
term.drawImage(imagePath);

async function loadImageAndPreprocess(imagePath) {
  // Read the image file using fs
  const imageData = fs.readFileSync(imagePath);

  // Decode the JPEG image
  const rawImageData = jpeg.decode(imageData, true); // true for buffer output

  // Convert the image data to a TensorFlow tensor
  const tensor = tf.tensor3d(rawImageData.data, [rawImageData.height, rawImageData.width, 4], 'int32');
  
  // Normalize the tensor values to the range [0, 1]
  const normalizedTensor = tensor.div(tf.scalar(255));

  // Reshape the tensor to [1, height, width, channels] (add batch dimension)
  const reshapedTensor = normalizedTensor.expandDims(0);

  return reshapedTensor;
}

// load the image and make the prediction
loadImageAndPreprocess(imagePath).then((imageTensor) => {
  // Now 'imageTensor' contains the preprocessed image data
  console.log('Image Tensor:', imageTensor);
  
  // iterate over data
  const arr = imageTensor.arraySync();
  console.log(arr[0]);
  
  arr[0].forEach((row, index) => {
    //if (index % 28 === 0)
    //  console.log('\n');
    console.log(`[${row}] ${index} \n`);
  });
  

  
  
});

*/


