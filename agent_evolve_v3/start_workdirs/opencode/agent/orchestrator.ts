import { createOpencodeClient, createOpencodeServer, type OpencodeClient } from "@opencode-ai/sdk/v2"
import { writeFileSync } from "fs"

const model = process.env.OPENCODE_MODEL
if (!model) {
  console.error("OPENCODE_MODEL is required")
  process.exit(1)
}

const instruction = process.argv.slice(2).join(" ")
if (!instruction) {
  console.error("Usage: bun run orchestrator.ts <instruction>")
  process.exit(1)
}

const trajectoryPath = process.env.OPENCODE_TRAJECTORY_PATH || ""
const port = parseInt(process.env.OPENCODE_PORT || "0") || 0

function buildProviderConfig(modelStr: string): Record<string, any> {
  const config: Record<string, any> = {}
  const [providerID, modelID] = modelStr.includes("/")
    ? [modelStr.split("/")[0], modelStr.split("/").slice(1).join("/")]
    : ["openai", modelStr]

  if (providerID === "openai") {
    config.openai = { models: { [modelID!]: {} } }
  }

  return config
}

async function connectEventStream(serverUrl: string): Promise<ReadableStreamDefaultReader<Uint8Array>> {
  for (let attempt = 0; attempt < 10; attempt++) {
    try {
      const resp = await fetch(`${serverUrl}/event`, {
        headers: { Accept: "text/event-stream" },
      })
      return resp.body!.getReader()
    } catch (e) {
      if (attempt === 9) throw e
      await Bun.sleep(1000 * (attempt + 1))
    }
  }

  throw new Error("Failed to connect to event stream")
}

async function waitForCompletion(
  reader: ReadableStreamDefaultReader<Uint8Array>,
  client: OpencodeClient,
  sessionID: string,
): Promise<void> {
  const decoder = new TextDecoder()
  const deadline = Date.now() + 600_000

  while (Date.now() < deadline) {
    const { value } = await Promise.race([
      reader.read(),
      new Promise<{ value: undefined; done: true }>((r) =>
        setTimeout(() => r({ value: undefined, done: true }), 5000),
      ),
    ])
    if (!value) continue

    for (const line of decoder.decode(value).split("\n")) {
      if (!line.startsWith("data:")) continue
      try {
        const evt = JSON.parse(line.slice(5).trim())
        if (evt.type === "permission.asked") {
          const reqID = evt.properties?.id
          if (reqID) {
            client.permission.reply({ requestID: reqID, reply: "always" }).catch(() => {})
          }
          continue
        }
        if (evt.properties?.sessionID !== sessionID) continue
        if (evt.type === "session.idle") return
        if (evt.type === "session.error") {
          console.error("Session error:", evt.properties?.error?.data?.message || "unknown error")
          return
        }
      } catch {}
    }
  }
}

async function extractTrajectory(client: OpencodeClient, sessionID: string): Promise<any[]> {
  const messages = await client.session.messages({ sessionID })
  return (messages.data ?? []).map((m: any) => ({
    role: m.info.role,
    agent: (m.info as any).agent,
    model: (m.info as any).modelID,
    text: m.parts
      .filter((p: any) => p.type === "text" && "text" in p)
      .map((p: any) => (p as any).text)
      .join(""),
    tools: m.parts
      .filter((p: any) => p.type === "tool")
      .map((p: any) => ({
        tool: (p as any).tool,
        status: (p as any).state?.status,
        input: (p as any).state?.input,
        output: (p as any).state?.output?.slice(0, 500),
      })),
    tokens: (m.info as any).tokens,
    cost: (m.info as any).cost,
  }))
}

async function run() {
  const server = await createOpencodeServer({
    ...(port > 0 ? { port } : {}),
    timeout: 60_000,
    config: {
      model: model!,
      snapshot: false,
      permission: "allow",
      provider: buildProviderConfig(model!),
    },
  })

  try {
    const client = createOpencodeClient({ baseUrl: server.url })
    const sessionID = (await client.session.create()).data!.id
    const reader = await connectEventStream(server.url)

    await client.session.promptAsync({
      sessionID,
      agent: "build",
      parts: [{ type: "text", text: instruction }],
    })

    await waitForCompletion(reader, client, sessionID)
    reader.cancel().catch(() => {})

    const trajectory = await extractTrajectory(client, sessionID)

    if (trajectoryPath) {
      writeFileSync(trajectoryPath, JSON.stringify(trajectory, null, 2))
    }

    console.log(JSON.stringify(trajectory))
  } finally {
    server.close()
  }
}

for (let serverAttempt = 0; serverAttempt < 5; serverAttempt++) {
  try {
    await run()
    break
  } catch (e: any) {
    const msg = e?.message ?? String(e)
    if (
      serverAttempt < 4 &&
      (msg.includes("ConnectionRefused") ||
        msg.includes("Unable to connect") ||
        msg.includes("timed out") ||
        msg.includes("TimeoutError"))
    ) {
      console.error(`Attempt ${serverAttempt + 1} failed, retrying: ${msg}`)
      await Bun.sleep(3000 * (serverAttempt + 1))
      continue
    }
    throw e
  }
}
