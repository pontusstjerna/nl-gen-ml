import fs from "fs"
import path from "path"
import * as tf from "@tensorflow/tfjs-node"

let vocabulary = []

// Number of tokens
export const sequenceLength = 6

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

    const inputTensor = tf.oneHot(nGrams, vocLen())
    const labelTensor = tf.oneHot(nextTokens, vocLen())

    return { inputs: inputTensor, labels: labelTensor }
  })
}

const createNgrams = allTokens => {
  let nGrams = []

  // Create nGrams from words with n = sequenceLength. This will be the inputs.
  for (let i = 0; i < allTokens.length - sequenceLength; i++) {
    nGrams.push(allTokens.slice(i, i + sequenceLength))
  }

  return nGrams
}

const createNextTokens = allTokens => {
  let nextTokens = []

  for (let i = 0; i < allTokens.length - sequenceLength; i++) {
    nextTokens.push(allTokens[i + sequenceLength])
  }

  return nextTokens
}

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
  unformatted.join(" ").replace(/ \./g, ".")

export const createTrainingData = async filePath => {
  const rawText = await loadFileData(filePath)

  // Split on spaces and newlines
  const allTokens = rawText
    .replace(/\./g, " .")
    .split(/ /g)
    //.flatMap(w => w.split(/\n/g))
    .filter(s => s.length > 0)
    .slice(0, 50000)

  console.log(allTokens)
  vocabulary = [...new Set(allTokens)]

  const nGrams = createNgrams(allTokens)
  const nextTokens = createNextTokens(allTokens)

  console.log(
    `Data formatted. 
    Number of tokens: ${vocLen()}
    Number of nGrams: ${nGrams.length}
    Sequence length: ${sequenceLength}`
  )

  const encodedNgrams = encodeNgrams(nGrams)

  const encodedNextTokens = nextTokens.map(token2ind)

  const trainingData = encodedNgrams.map((nGram, index) => ({
    nGram,
    nextToken: encodedNextTokens[index],
  }))

  const trainingTensors = convertToTensors(trainingData)

  return trainingTensors
}
