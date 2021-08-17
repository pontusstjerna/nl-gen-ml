import * as tf from "@tensorflow/tfjs-node-gpu"
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
  batchesPerEpoch,
} from "./dataHandler"

const defaultGenerationTokenCount = 100
const layerSize = 128

const createModel = (nInputs, nOutputs) => {
  const model = tf.sequential()

  // Input layer
  model.add(
    tf.layers.lstm({
      inputShape: [sequenceLength(), nInputs],
      units: layerSize, // Number of neurons
      //returnSequences: true,
    })
  )

  //model.add(tf.layers.dropout(0.3))

  // Hidden
  /*model.add(
    tf.layers.lstm({
      units: 256,
    })
  )

  model.add(
    tf.layers.dense({
      units: 256,
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

const trainModel = async (model, dataGenerator, batchSize, epochs = 100) => {
  model.compile({
    optimizer: tf.train.adam(),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  })

  const dataset = tf.data.generator(dataGenerator).repeat(epochs)

  return await model.fitDataset(dataset, {
    batchesPerEpoch: batchesPerEpoch(batchSize),
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

const generate = (model, initialInput = "Recept på", temperature = 1) => {
  const seqLength = sequenceLength(model)
  const generationTokenCount =
    process.env.generationCount || defaultGenerationTokenCount

  let inputNgram = initialInput.split(/ /g).slice(0, seqLength)
  let output = inputNgram

  if (inputNgram.length < seqLength) {
    inputNgram = [
      ...new Array(seqLength - inputNgram.length).fill(null),
      ...inputNgram,
    ]
  }

  for (let i = 0; i < generationTokenCount; i++) {
    tf.tidy(() => {
      let input = nGram2oneHot(inputNgram)

      const outputTensor = model.predict(input).reshape([vocLen()])
      const outputToken = oneHot2token(outputTensor, temperature)

      output = [...output, outputToken]

      inputNgram = [...inputNgram.slice(1, inputNgram.length), outputToken]
    })
  }

  return output
}

export default async (
  train,
  modelName = "model",
  initializer,
  dataFilePath,
  epochs,
  temperature
) => {
  let model = null

  if (modelName) {
    try {
      model = await tf.loadLayersModel(`file://models/${modelName}/model.json`)
      console.log(`Loaded model models/${modelName}.json`)
    } catch (error) {
      console.log(
        `Model ${modelName} doesn't exist, creating and training a new one.`
      )
      train = true
    }
  }

  if (train) {
    await loadTrainingData(dataFilePath, sequenceLength(model))
    const numTokens = vocLen()
    model = model || createModel(numTokens, numTokens)
    const batchSize = parseInt(process.env.batchSize) || 128

    await trainModel(
      model,
      trainingDataGenerator.bind(null, batchSize, sequenceLength(model)),
      batchSize,
      epochs
    )

    console.log(`Training done.`)
    await model.save(`file://models/${modelName}`)
    console.log(`Model saved as "${modelName}.json"`)
    saveVocabulary(modelName)
  } else {
    loadVocabulary(modelName)
  }

  return formatOutput(generate(model, initializer, temperature))
}
