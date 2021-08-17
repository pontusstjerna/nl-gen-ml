import fs from "fs"
import path from "path"
import * as tf from "@tensorflow/tfjs-node-gpu"

let vocabulary = []
let unfrequentTokens = []
let nGrams = []
let nextTokens = []

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

const filterTokensByFrequency = (allTokens, vocabulary) => {
  const minFreq = process.env.minTokenFrequency || 1
  let filteredVocab = []
  let removedTokens = []
  console.log(
    `Filtering ${vocabulary.length} tokens with frequency < ${minFreq}\n`
  )

  for (let i = 0; i < vocabulary.length; i++) {
    const tokenFrequency = allTokens.reduce(
      (count, token) => (vocabulary[i] === token ? count + 1 : count),
      0
    )

    if (tokenFrequency < minFreq) {
      removedTokens.push(vocabulary[i])
    } else {
      filteredVocab.push(vocabulary[i])
    }

    const percent = Math.floor(vocabulary.length / 100)
    if (i % percent === 0) {
      process.stdout.write(`${Math.floor(i / percent)}%...\r`)
    }
  }

  console.log(
    `Filtered ${removedTokens.length} tokens with frequency < ${minFreq}`
  )

  return { filtered: filteredVocab, removed: removedTokens }
}

const capitalize = string => string.charAt(0).toUpperCase() + string.slice(1)

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

export const formatOutput = tokenArr => {
  const unformatted = tokenArr.join(" ").replace(/_/g, "")
  const spacedSeparationTokens = unformatted.match(spacedSeparationTokenRegexp)
  return spacedSeparationTokens
    .reduce(
      (result, separationToken) =>
        result.replaceAll(` ${separationToken}`, separationToken),
      unformatted
    )
    .split(". ")
    .map(capitalize)
    .join(". ")
}

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
  const allTokens = rawText
    .replace(separationTokenRegexp, " $& ")
    .replace(spacedSeparationTokenRegexp, " $&_ ")
    .toLowerCase()
    .split(/[ ]+/g)

  const { filtered, removed } = filterTokensByFrequency(allTokens, [
    ...new Set(allTokens),
  ])
  vocabulary = filtered
  unfrequentTokens = removed

  // Remove nGrams that has any removed token in it
  const sequences = createNgrams(allTokens, sequenceLength + 1).filter(
    nGram => !nGram.some(token => unfrequentTokens.includes(token))
  )

  nGrams = sequences.map(nGram => nGram.slice(0, sequenceLength))
  nextTokens = sequences.map(nGram => nGram[sequenceLength])

  console.log(
    `Data formatted. 
Number of vocabulary: ${vocLen()}
Sequence length: ${sequenceLength}
Number of nGrams: ${nGrams.length}`
  )
}

export const batchesPerEpoch = batchSize =>
  Math.min(
    Math.floor(nGrams.length / batchSize),
    parseInt(process.env.batchCountLimit) || 1000000
  )

export function* trainingDataGenerator(batchSize, sequenceLength) {
  const numberOfBatches = batchesPerEpoch(batchSize)

  console.log(`Batch size: ${batchSize}
Number of batches: ${numberOfBatches}`)

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
