import fs from "fs"
import path from "path"
import * as tf from "@tensorflow/tfjs"

let vocabulary = []

const loadFileData = filePath =>
  new Promise((resolve, reject) => {
    const rawText = fs
      .readFileSync(path.join(process.cwd(), filePath))
      .toString("utf8")

    resolve(rawText.substring(0, 100000))
  })

const convertToTensor = data => {
  return tf.tidy(() => {
    // tf.util.shuffle(data)

    const inputs = data.map(({ current }) => current)
    const labels = data.map(({ next }) => next)

    const inputTensor = tf.oneHot(inputs, vocabulary.length)
    const labelTensor = tf.oneHot(labels, vocabulary.length)

    // inputTensor.print()
    // labelTensor.print()

    return { inputs: inputTensor, labels: labelTensor }
  })
}

export const createTrainingData = async filePath => {
  const rawText = await loadFileData(filePath)

  // Split on spaces and newlines
  const allWords = rawText.split(/ /g).flatMap(w => w.split(/\n/g))
  vocabulary = [...new Set(allWords)]

  const trainingData = allWords
    .map((word, index) => ({ current: word, next: allWords[index + 1] }))
    .slice(0, allWords.length - 1)

  // Use vocabulary indexes to encode to numbers
  const encodedTrainingData = trainingData.map(entry => ({
    current: vocabulary.indexOf(entry.current),
    next: vocabulary.indexOf(entry.next),
  }))

  return convertToTensor(encodedTrainingData)
}
