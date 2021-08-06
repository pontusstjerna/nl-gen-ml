import * as tf from "@tensorflow/tfjs-node"
import {
  createTrainingData,
  nGram2oneHot,
  oneHot2token,
  sequenceLength,
  vocLen,
} from "./dataHandler"

const generationTokenCount = 20

const trainModel = (model, inputs, labels) => {
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.softmaxCrossEntropy,
    metrics: ["mse"],
  })

  const batchSize = 32
  const epochs = 10

  return model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: [
      {
        onEpochEnd: (epoch, logs) => {
          console.log(logs)
        },
      },
    ],
  })
}

const createModel = (nInputs, nOutputs) => {
  const model = tf.sequential()

  // Input layer
  model.add(
    tf.layers.lstm({
      inputShape: [sequenceLength, nInputs],
      units: 128, // Number of neurons
      useBias: true,
      activation: "tanh",
    })
  )

  /*// Hidden
  model.add(
    tf.layers.dense({
      units: nInputs * 2,
      useBias: true,
      activation: "tanh",
    })
  )*/

  // Output layer
  model.add(
    tf.layers.dense({
      units: nOutputs,
      useBias: true,
      activation: "softmax",
    })
  )

  return model
}

const generate = (model, initialInput = "Recept   ") => {
  let inputNgram = initialInput.split(/ /g).slice(0, sequenceLength)
  let output = inputNgram

  if (inputNgram.length < sequenceLength) {
    inputNgram = [
      ...new Array(sequenceLength - inputNgram.length).fill(null),
      ...inputNgram,
    ]
  }

  for (let i = 0; i < generationTokenCount; i++) {
    let input = nGram2oneHot(inputNgram)

    const outputTensor = model.predict(input).reshape([vocLen()])
    const outputToken = oneHot2token(outputTensor)
    output = [...output, outputToken]

    inputNgram = [...inputNgram.slice(1, inputNgram.length), outputToken]

    input = nGram2oneHot(inputNgram)
  }

  console.log(output.join(" "))

  return output
}

export default async () => {
  const trainingData = await createTrainingData("../ml-data/recipes.txt")

  const numTokens = vocLen()
  const model = createModel(numTokens, numTokens)

  await trainModel(model, trainingData.inputs, trainingData.labels)
  console.log("Training done! Will try and predict some.")

  return generate(model)
}
