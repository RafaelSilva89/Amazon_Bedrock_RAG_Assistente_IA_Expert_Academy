
import csv
from pathlib import Path

import boto3
import streamlit as st
from botocore.exceptions import BotoCoreError, ClientError

# ------------------------------------------------------------------
# 1. Configurações (as mesmas do notebook de RAG)
# ------------------------------------------------------------------
KNOWLEDGE_BASE_ID = "4SCLDZ9ZTH"
MODEL_ID = "us.amazon.nova-micro-v1:0"
REGION = "us-east-1"
# O CSV de credenciais fica na mesma pasta do app.py (fora do código)
ARQUIVO_CREDENCIAIS = Path(__file__).parent / "curso_bedrock_accessKeys.csv"

st.set_page_config(
    page_title="Assistente IA Expert Academy",
    page_icon="🤖",
    layout="wide"
)


# ------------------------------------------------------------------
# 2. Conexão com a AWS
# ------------------------------------------------------------------
# O Streamlit reexecuta o script a cada interação. O cache_resource
# faz os clientes serem criados uma única vez e reaproveitados.
@st.cache_resource
def criar_clientes():
    with open(ARQUIVO_CREDENCIAIS, encoding="utf-8-sig") as f:
        credenciais = next(csv.DictReader(f))

    session = boto3.Session(
        aws_access_key_id=credenciais["Access key ID"],
        aws_secret_access_key=credenciais["Secret access key"],
        region_name=REGION,
    )
    # bedrock-agent-runtime: busca na Knowledge Base | bedrock-runtime: geração da resposta
    return session.client("bedrock-agent-runtime"), session.client("bedrock-runtime")


# ------------------------------------------------------------------
# 3. RAG em duas etapas: busca + geração
# ------------------------------------------------------------------
def ask(question, n_resultados=5, temperature=0.2):
    bedrock_agent_runtime, bedrock_runtime = criar_clientes()

    # Retrieval: a base é "Bedrock Managed", por isso usa managedSearchConfiguration
    retrieval = bedrock_agent_runtime.retrieve(
        knowledgeBaseId=KNOWLEDGE_BASE_ID,
        retrievalQuery={"text": question},
        retrievalConfiguration={
            "managedSearchConfiguration": {"numberOfResults": n_resultados}
        }
    )
    resultados = retrieval["retrievalResults"]

    # Sem trechos encontrados não há contexto, então nem chamamos o modelo
    if not resultados:
        return {
            "resposta": "Não encontrei informações sobre isso na base de conhecimento.",
            "fontes": []
        }

    contexto = "\n\n".join(r["content"]["text"] for r in resultados)

    # Generation: o modelo responde usando apenas o contexto recuperado
    prompt = (
        "Responda à pergunta usando somente as informações do contexto abaixo. "
        "Se a resposta não estiver no contexto, diga que não sabe.\n\n"
        f"Contexto:\n{contexto}\n\nPergunta: {question}"
    )
    response = bedrock_runtime.converse(
        modelId=MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 512, "temperature": temperature}
    )

    return {
        "resposta": response["output"]["message"]["content"][0]["text"],
        "fontes": resultados
    }


# ------------------------------------------------------------------
# 4. Exibição das fontes (citações)
# ------------------------------------------------------------------
def mostrar_fontes(fontes):
    if not fontes:
        return

    with st.expander(f"📄 Fontes ({len(fontes)} trechos)"):
        for i, fonte in enumerate(fontes, start=1):
            uri = (fonte.get("location") or {}).get("s3Location", {}).get("uri", "")
            arquivo = uri.rsplit("/", 1)[-1] if uri else "documento"
            pagina = (fonte.get("metadata") or {}).get("x-amz-bedrock-kb-document-page-number")
            score = fonte.get("score")

            titulo = f"**Trecho {i}** · {arquivo}"
            if pagina is not None:
                titulo += f" · página {int(float(pagina))}"
            if score is not None:
                titulo += f" · score {score:.3f}"

            st.markdown(titulo)
            st.caption(fonte["content"]["text"])


# ------------------------------------------------------------------
# 5. Interface
# ------------------------------------------------------------------
st.title("🤖 Assistente IA Expert Academy")
st.caption("Pergunte sobre cursos, trilhas, planos e termos de uso da IA Expert Academy.")

# O histórico fica no session_state para sobreviver às reexecuções do script
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("⚙️ Configurações")
    n_resultados = st.slider("Trechos buscados na base", min_value=1, max_value=10, value=5)
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1)

    if st.button("🗑️ Limpar conversa"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption(f"Base de conhecimento: `{KNOWLEDGE_BASE_ID}`  \nModelo: `{MODEL_ID}`")

# Reexibe o histórico da conversa
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        mostrar_fontes(message.get("fontes"))

if prompt := st.chat_input("Digite sua pergunta"):
    # Usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Assistente
    with st.chat_message("assistant"):
        with st.spinner("Consultando a base de conhecimento..."):
            try:
                resultado = ask(prompt, n_resultados=n_resultados, temperature=temperature)
            except FileNotFoundError:
                st.error(f"Arquivo de credenciais não encontrado: `{ARQUIVO_CREDENCIAIS.name}`")
            except (ClientError, BotoCoreError) as erro:
                st.error(f"Erro ao consultar a AWS: {erro}")
            else:
                st.markdown(resultado["resposta"])
                mostrar_fontes(resultado["fontes"])
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": resultado["resposta"],
                    "fontes": resultado["fontes"]
                })
