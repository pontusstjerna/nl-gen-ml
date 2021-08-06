import * as tf from "@tensorflow/tfjs-node"
import {
  createTrainingData,
  fromOneHotToToken,
  fromTokenToOneHot,
  getVocabulary,
} from "./dataHandler"

const trainModel = (model, inputs, labels) => {
  model.compile({
    optimizer: "sgd",
    loss: tf.losses.meanSquaredError,
    metrics: ["mse"],
  })

  const batchSize = 32
  const epochs = 50

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
    tf.layers.dense({
      inputShape: [nInputs],
      units: nInputs,
      useBias: true,
      activation: "tanh",
    })
  )

  // Hidden
  model.add(
    tf.layers.dense({
      units: nInputs * 2,
      useBias: true,
      activation: "tanh",
    })
  )

  // Output layer
  model.add(
    tf.layers.dense({
      units: nOutputs,
      useBias: true,
      activation: "sigmoid",
    })
  )

  return model
}

const predictToken = (model, token) => {
  const input = fromTokenToOneHot(token)
  const output = model.predict(input)

  const predictedToken = fromOneHotToToken(output)
  return predictedToken
}

const demo = model => {
  const allTokens = getVocabulary()

  let inputToken = allTokens[Math.floor(Math.random() * (allTokens.length - 1))]

  let output = inputToken

  for (let i = 0; i < 100; i++) {
    let predictedToken = predictToken(model, inputToken)
    output += ` ${predictedToken}`
    inputToken = predictedToken
  }

  console.log(output)
}

export default async () => {
  const trainingData = await createTrainingData("../ml-data/recipes.txt")

  //trainingData.inputs.print()
  //trainingData.labels.print()
  /*const numTokens = getVocabulary().length
  const model = createModel(numTokens, numTokens)

  await trainModel(model, trainingData.inputs, trainingData.labels)
  console.log("Training done! Will try and predict some.")

  demo(model)

  return "hello"*/
}
