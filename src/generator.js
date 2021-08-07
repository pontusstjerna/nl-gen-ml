import * as tf from "@tensorflow/tfjs-node"
import { truncatedNormal } from "@tensorflow/tfjs-node"
import {
  createTrainingData,
  formatOutput,
  nGram2oneHot,
  oneHot2token,
  sequenceLength,
  vocLen,
} from "./dataHandler"

const generationTokenCount = 200

const trainModel = (model, inputs, labels) => {
  model.compile({
    optimizer: tf.train.adam(), //tf.train.rmsprop(0.1),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  })

  const batchSize = 128
  const epochs = 100

  return model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: [
      {
        onEpochEnd: (epoch, logs) => {
          console.log(
            `Epoch ${epoch + 1} finished with an accuracy of ${
              Math.round(logs.acc * 10000) / 100
            }%`
          )
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
      units: 100, // Number of neurons
      //returnSequences: true,
    })
  )

  //model.add(tf.layers.dropout(0.3))

  // Hidden
  /*model.add(
    tf.layers.lstm({
      units: 100,
    })
  )

  model.add(
    tf.layers.dense({
      units: 100,
      useBias: true,
      activation: "relu",
    })
  )*/

  // Output layer
  model.add(
    tf.layers.dense({
      units: nOutputs,
      //useBias: true,
      activation: "softmax",
    })
  )

  return model
}

const generate = (model, initialInput = "Recept på") => {
  let inputNgram = initialInput.split(/ /g).slice(0, sequenceLength)
  let output = inputNgram

  if (inputNgram.length < sequenceLength) {
    inputNgram = [
      ...new Array(sequenceLength - inputNgram.length).fill(null),
      ...inputNgram,
    ]
  }

  for (let i = 0; i < generationTokenCount; i++) {
    tf.tidy(() => {
      let input = nGram2oneHot(inputNgram)

      const outputTensor = model.predict(input).reshape([vocLen()])
      const outputToken = oneHot2token(outputTensor)

      output = [...output, outputToken]

      inputNgram = [...inputNgram.slice(1, inputNgram.length), outputToken]
    })
  }

  return output
}

export default async () => {
  const trainingData = await createTrainingData("../ml-data/recipes.txt")

  const numTokens = vocLen()
  const model = createModel(numTokens, numTokens)

  await trainModel(model, trainingData.inputs, trainingData.labels)
  trainingData.inputs.dispose()
  trainingData.labels.dispose()

  console.log("Training done! Will try and predict some.")

  return formatOutput(generate(model))
}
