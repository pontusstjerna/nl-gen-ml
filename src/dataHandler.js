import fs from "fs"
import path from "path"
import * as tf from "@tensorflow/tfjs-node"
import { start } from "repl"

let vocabulary = []
let allTokens = []

// Number of tokens
export const sequenceLength = () => parseInt(process.env.sequenceLength) || 4

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

const createNgrams = allTokens =>
  new Array(allTokens.length - sequenceLength())
    .fill()
    .map((_, startIndex) =>
      allTokens.slice(startIndex, startIndex + sequenceLength())
    )

const createNextTokens = allTokens => allTokens.slice(sequenceLength())

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
  unformatted.join(" ").replace(/ \./g, ".").replace(/ \n /g, "\n")

export const saveVocabulary = modelName => {
  try {
    fs.writeFileSync(
      path.join(process.cwd(), `models/${modelName}/vocab.txt`),
      JSON.stringify(vocabulary)
    )
    console.log(`Vocabulary saved with ${vocLen()} tokens`)
  } catch (error) {
    console.log("Unable to save vocabulary.")
  }
}

export const loadVocabulary = modelName => {
  const json = fs
    .readFileSync(path.join(process.cwd(), `models/${modelName}/vocab.txt`))
    .toString("utf8")
  vocabulary = JSON.parse(json)
}

export const loadTrainingData = async filePath => {
  const rawText = await loadFileData(filePath)

  // Split on spaces and newlines
  allTokens = rawText
    .replace(/\./g, " .")
    .replace(/\n/g, " \n ")
    .split(/ /g)
    //.flatMap(w => w.split(/\n/g))
    .filter(s => s.length > 0)
    .slice(0, 10)

  vocabulary = [...new Set(allTokens)]

  console.log(
    `Data formatted. 
    Number of vocabulary: ${vocLen()}
    Number of all tokens: ${allTokens.length}
    Sequence length: ${sequenceLength()}`
  )
}

export function* trainingDataGenerator() {
  const chunkSize = 4
  for (let i = 0; i < Math.ceil(allTokens.length / chunkSize); i++) {
    const chunkedTokens = allTokens.slice(i, i + chunkSize + sequenceLength())

    const nGrams = createNgrams(chunkedTokens)
    const nextTokens = createNextTokens(chunkedTokens)

    const encodedNgrams = encodeNgrams(nGrams)
    const encodedNextTokens = nextTokens.map(token2ind)

    const trainingData = encodedNgrams.map((nGram, index) => ({
      nGram,
      nextToken: encodedNextTokens[index],
    }))

    yield convertToTensors(trainingData)
  }
}
