import fs from "fs"
import path from "path"
import * as tf from "@tensorflow/tfjs-node-gpu"

let vocabulary = []
let allTokens = []

const separationTokenRegexp = /[(\-–\n]/g
const spacedSeparationTokenRegexp = /[.,):!?]/g

// Number of tokens
export const sequenceLength = model =>
  model
    ? model.getConfig().layers[0].config.batchInputShape[1] // Sequence length of the saved model is saved in the second dimension of the input shape
    : parseInt(process.env.sequenceLength) || 4

const loadFileData = filePath =>
  new Promise((resolve, reject) => {
    const rawText = fs
      .readFileSync(path.join(process.cwd(), filePath))
      .toString("utf8")

    resolve(rawText)
  })

const convertToTensors = data => {
  return tf.tidy(() => {
    tf.util.shuffle(data)

    const nGrams = data.map(({ nGram }) => nGram)
    const nextTokens = data.map(({ nextToken }) => nextToken)

    const inputTensor = tf.oneHot(nGrams, vocLen()).cast("float32")
    const labelTensor = tf.oneHot(nextTokens, vocLen()).cast("float32")

    return { xs: inputTensor, ys: labelTensor }
  })
}

const createNgrams = (tokens, sequenceLength) =>
  new Array(tokens.length - sequenceLength)
    .fill()
    .map((_, startIndex) =>
      tokens.slice(startIndex, startIndex + sequenceLength)
    )

const encodeNgram = nGram => nGram.map(token2ind)

/**
 * Encoding example
 * Consider the nGram ["hello", "there", "general", "kenobi"]
 * And our vocabulary is ["general", "kenobi", "hello", "there"]
 * That would be encoded:
 * [2, 3, 0, 1]
 * And a list of encoded engrams would simply be [[2, 3, 0, 1], ...]
 * But since we can't use numbers as inputs to our RNN, we need to one-hot encode the numbers as well, like so:
 * [[[0,0,1,0], [0,0,0,1], [1,0,0,0], [0,1,0,0]]]
 */
const encodeNgrams = nGrams => nGrams.map(encodeNgram)

export const token2ind = token => vocabulary.indexOf(token)

export const ind2token = index => vocabulary[index]

export const vocLen = () => vocabulary.length

export const nGram2oneHot = nGram => tf.oneHot([encodeNgram(nGram)], vocLen())

export const oneHot2token = tensor => {
  // Tensor is on format [0, 0, 0, 0, 0.7, 0.3, 0, 0] which should be translated to 4 etc
  const index = tf.argMax(tensor).arraySync()
  return ind2token(index)
}

export const formatOutput = unformatted =>
  unformatted.join(" ").replace(/_/g, " ")

export const saveVocabulary = modelName => {
  try {
    fs.writeFileSync(
      path.join(process.cwd(), `models/${modelName}/vocab.json`),
      JSON.stringify(vocabulary)
    )
    console.log(`Vocabulary saved with ${vocLen()} tokens`)
  } catch (error) {
    console.log("Unable to save vocabulary.")
  }
}

export const loadVocabulary = modelName => {
  const json = fs
    .readFileSync(path.join(process.cwd(), `models/${modelName}/vocab.json`))
    .toString("utf8")
  vocabulary = JSON.parse(json)
}

export const loadTrainingData = async (
  filePath,
  sequenceLength = sequenceLength()
) => {
  const rawText = await loadFileData(filePath)

  // Split on spaces and newlines
  allTokens = rawText
    .replace(separationTokenRegexp, " $& ")
    .replace(spacedSeparationTokenRegexp, " $&_ ")
    .toLowerCase()
    .split(/[ ]+/g)
  //.flatMap(w => w.split(/\n/g))
  //.slice(0, 50)

  vocabulary = [...new Set(allTokens)]

  console.log(
    `Data formatted. 
Number of vocabulary: ${vocLen()}
Number of all tokens: ${allTokens.length}
Sequence length: ${sequenceLength}`
  )
}

export const batchesPerEpoch = batchSize =>
  Math.min(
    Math.floor(allTokens.length / batchSize),
    parseInt(process.env.batchCountLimit) || 1000000
  )

export function* trainingDataGenerator(batchSize, sequenceLength) {
  const numberOfBatches = batchesPerEpoch(batchSize)

  const nGrams = createNgrams(allTokens, sequenceLength)
  const nextTokens = allTokens.slice(sequenceLength)

  console.log(
    `Batch size: ${batchSize}
Number of batches: ${numberOfBatches}
Number of nGrams: ${nGrams.length}`
  )

  const encodedNgrams = encodeNgrams(nGrams)
  const encodedNextTokens = nextTokens.map(token2ind)

  for (let i = 0; i < numberOfBatches; i++) {
    const nGramsInBatch = encodedNgrams.slice(
      i * batchSize,
      i * batchSize + batchSize
    )
    const nextTokensInBatch = encodedNextTokens.slice(
      i * batchSize,
      i * batchSize + batchSize
    )

    const trainingData = nGramsInBatch.map((nGram, index) => ({
      nGram,
      nextToken: nextTokensInBatch[index],
    }))

    yield convertToTensors(trainingData)
  }
}
