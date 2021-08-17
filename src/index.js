import generate from "./generator"
import { config } from "dotenv"

config()

console.log(process.argv)

const parseArg = argName => {
  const arg = process.argv.find(x => x.startsWith(argName))
  return arg ? arg.split("=")[1] : undefined
}

const train =
  process.argv.some(arg => arg === "-T" || arg === "--train") ||
  process.env.train

const modelName = parseArg("--name") || process.env.name

const initializer = parseArg("--initializer") || process.env.initializer

const dataFilePath = parseArg("--dataFilePath") || process.env.dataFilePath

const epochs = parseArg("--epochs") || parseInt(process.env.epochs)

const temperature =
  parseFloat(parseArg("--temperature")) || parseFloat(process.env.temperature)

generate(train, modelName, initializer, dataFilePath, epochs, temperature).then(
  console.log
)
