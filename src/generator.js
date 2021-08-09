import * as tf from "@tensorflow/tfjs-node"
import {
  trainingDataGenerator,
  formatOutput,
  loadTrainingData,
  loadVocabulary,
  nGram2oneHot,
  oneHot2token,
  saveVocabulary,
  sequenceLength,
  vocLen,
} from "./dataHandler"

const defaultGenerationTokenCount = 200

const createModel = (nInputs, nOutputs) => {
  const model = tf.sequential()

  // Input layer
  model.add(
    tf.layers.lstm({
      inputShape: [sequenceLength(), nInputs],
      units: 100, // Number of neurons
      returnSequences: true,
    })
  )

  //model.add(tf.layers.dropout(0.3))

  // Hidden
  model.add(
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
  )

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

const trainModel = async (model, dataGenerator) => {
  model.compile({
    optimizer: tf.train.adam(), //tf.train.rmsprop(0.1),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  })

  const batchSize = parseInt(process.env.batchSize) || 128
  const epochs = parseInt(process.env.epochs) || 100

  const dataset = tf.data.generator(dataGenerator)

  return await model.fitDataset(dataset, {
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

const generate = (model, initialInput = "Recept på") => {
  // Sequence length of the saved model is saved in the second dimension of the input shape
  const sequenceLength = model.getConfig().layers[0].config.batchInputShape[1]
  const generationTokenCount =
    process.env.generationCount || defaultGenerationTokenCount

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

const createAndTrainModel = async trainingDataFilePath => {
  await loadTrainingData(trainingDataFilePath)

  const numTokens = vocLen()
  const model = createModel(numTokens, numTokens)

  await trainModel(model, trainingDataGenerator)

  console.log(`Training done.`)

  return model
}

export default async (
  train,
  modelName = "model",
  initializer,
  dataFilePath
) => {
  let model = null

  if (modelName) {
    // Try to load
    console.log(`Trying to load models/${modelName}/model.json`)
    try {
      model = await tf.loadLayersModel(`file://models/${modelName}/model.json`)
    } catch (error) {
      console.log(
        `Model ${modelName} doesn't exist, creating and training a new one.`
      )
      train = true
    }
  }

  if (train) {
    model = await createAndTrainModel(dataFilePath)
    await model.save(`file://models/${modelName}`)
    console.log(`Model saved as "${modelName}.json"`)
    saveVocabulary(modelName)
  } else {
    loadVocabulary(modelName)
  }

  return formatOutput(generate(model, initializer))
}
