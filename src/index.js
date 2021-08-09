import generate from "./generator"

console.log(process.argv)

const train = process.argv.some(arg => arg === "-T" || arg === "--train")

const modelName = (() => {
  const nameArg = process.argv.find(arg => arg.startsWith("--name"))
  return nameArg ? nameArg.split("=")[1] : undefined
})()

const initializer = (() => {
  const initializerArg = process.argv.find(arg =>
    arg.startsWith("--initializer")
  )
  return initializerArg
    ? initializerArg.split("=")[1].replace(/_/g, " ")
    : undefined
})()

const dataFilePath = (() => {
  const filePathArg = process.argv.find(arg => arg.startsWith("--dataFilePath"))
  return filePathArg ? filePathArg.split("=")[1] : undefined
})()

generate(train, modelName, initializer, dataFilePath).then(console.log)
