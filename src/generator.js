import * as tf from "@tensorflow/tfjs"
import { createTrainingData } from "./trainingDataHandler"

export default async () => {
  const trainingData = await createTrainingData("../ml-data/recipes.txt")

  console.log(trainingData)

  return "hello"
}
