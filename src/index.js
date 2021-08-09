import generate from "./generator"
import { config } from "dotenv"

config()

console.log(process.argv)

const train =
  process.argv.some(arg => arg === "-T" || arg === "--train") ||
  process.env.train

const modelName =
  (() => {
    const nameArg = process.argv.find(arg => arg.startsWith("--name"))
    return nameArg ? nameArg.split("=")[1] : undefined
  })() || process.env.name

const initializer =
  (() => {
    const initializerArg = process.argv.find(arg =>
      arg.startsWith("--initializer")
    )
    return initializerArg
      ? initializerArg.split("=")[1].replace(/_/g, " ")
      : undefined
  })() || process.env.initializer

const dataFilePath =
  (() => {
    const filePathArg = process.argv.find(arg =>
      arg.startsWith("--dataFilePath")
    )
    return filePathArg ? filePathArg.split("=")[1] : undefined
  })() || process.env.dataFilePath

generate(train, modelName, initializer, dataFilePath).then(console.log)
