# Azure AI Search: Cómo optimizar la búsqueda para tus aplicaciones RAG

## ¿Cómo funciona Azure AI Search?
[Azure AI Search](https://learn.microsoft.com/en-us/azure/search/search-what-is-azure-search) es un servicio de búsqueda de información que utiliza IA para mejorar la relevancia y precisión de los resultados.

Azure AI Search permite indexar información utilizando vectores, lo que significa que puede buscar no solo por palabras clave, sino también por conceptos y relaciones entre términos. Esto es especialmente útil para aplicaciones de [Retrival-Augmented Generation (RAG)](https://azure.microsoft.com/en-us/products/ai-services/ai-search).

## Ejemplo de un proceso de indexación

![Proceso de indexación](https://github.com/Azure-Samples/azure-search-openai-demo/raw/main/docs/images/diagram_prepdocs.png)

## ¿Qué es una indexación de vectores (vector embeddings)?

Un embedding vectorial es una representación numérica de un objeto, como un texto o una imagen, que captura sus características semánticas. En el contexto de Azure AI Search, los embeddings permiten que el motor de búsqueda comprenda mejor el significado y la relación entre diferentes términos.

**Ejemplo:** 

* "Perro" => [0.1, 0.2, 0.3, ...]
* "Animal" => [0.4, 0.5, 0.6, ...]

En este ejemplo, los vectores de "Perro" y "Animal" están cerca en el espacio vectorial, lo que indica que están relacionados semánticamente. Esto permite que Azure AI Search encuentre resultados relevantes incluso si las palabras exactas no coinciden.

## Generación de vector embeddings con OpenAI

### Modelos para generación de embeddings

- **text-embedding-ada-002**: Un modelo optimizado para tareas de embeddings, que combina un buen equilibrio entre velocidad y precisión.
- **text-embedding-3-small**: Un modelo de tamaño pequeño, ideal para aplicaciones que requieren una generación rápida de embeddings con un costo reducido.
- **text-embedding-3-large**: Un modelo de tamaño grande, que ofrece una mayor precisión en la generación de embeddings, pero a un costo más alto y con un tiempo de respuesta más lento.

### Ejemplo de generación de embeddings

```python
import os
import dotenv
import openai

dotenv.load_dotenv()

openai_client = openai.OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ["GITHUB_TOKEN"]
)
MODEL_NAME = "text-embedding-3-small"

content_input = "Hoja de vida: Lionel Messi. Futbolista argentino, considerado uno de los mejores jugadores de fútbol de todos los tiempos."

embeddings_response = openai_client.embeddings.create(
    model=MODEL_NAME,
    input=content_input,
)
embedding = embeddings_response.data[0].embedding

print(len(embedding))
print(embedding)
```
```
1536
[0.01875680685043335, -0.01207759603857994, -0.014209441840648651,...]
```

### Comparación de similitud entre vectores

Para comparar la similitud entre dos vectores, se puede utilizar una métrica de distancia como la "distancia coseno". Esta métrica mide el ángulo entre dos vectores y es útil para determinar cuán similares son en términos de dirección.

**Por ejemplo:** Supongamos que queremos comparar los vectores del texto generado conteriormente con una frase de consulta como "Diego Zumárraga Mera". Podemos calcular la similitud entre los dos vectores generados por OpenAI.

Primero generamos los embeddings para ambas frases

```python
content_input = "Hoja de vida: Lionel Messi. Futbolista argentino, considerado uno de los mejores jugadores de fútbol de todos los tiempos."

embeddings_response = openai_client.embeddings.create(
    model=MODEL_NAME,
    input=content_input,
)
content_embedding = embeddings_response.data[0].embedding
print(content_embedding)

embeddings_response = openai_client.embeddings.create(
    model=MODEL_NAME,
    input="Diego Zumárraga Mera",
)
query_embedding1 = embeddings_response.data[0].embedding
print(query_embedding1)
```

Luego, calculamos la distancia coseno entre los dos vectores:

```python
def cosine_similarity(v1, v2):

  dot_product = sum(
    [a * b for a, b in zip(v1, v2)])
  
  magnitude = (
    sum([a**2 for a in v1]) *
    sum([a**2 for a in v2])) ** 0.5

  return dot_product / magnitude

# Compare the two vectors
similarity = cosine_similarity(query_embedding1, content_embedding)
print(f"Similarity: {similarity:.4f}")
```
```
Similarity: 0.1799
```

Como podemos ver, la similitud entre los dos vectores es de aproximadamente 0.1799, lo que indica que no son muy similares semánticamente.

Ahora cambiemos la frase de consulta a "Hoja de Vida: Diego Zumárraga Mera" y volvamos a calcular la similitud:

```python
embeddings_response = openai_client.embeddings.create(
    model=MODEL_NAME,
    input="Resume la hoja de vida de Diego Zumárraga Mera",
)
query_embedding2 = embeddings_response.data[0].embedding
print(query_embedding2)

# Compare the two vectors
similarity = cosine_similarity(query_embedding2, content_embedding)
print(f"Similarity: {similarity:.4f}")
```
```
Similarity: 0.3429
```
Código completo en el notebook: [VectorEmbeddings.ipynb](VectorEmbeddings.ipynb)

La similitud entre los dos vectores aumentó a casi el doble, lo que indica que ahora son más similares semánticamente. Esto es porque la frase "Hoja de Vida" genera una similitud con cualquier texto que contenga esa frase.

### Conlusión
Si queremos desarrollar una solución RAG para indexar y buscar documentos como hojas de vida, esta puede tener problemas para buscar una hoja de vida específica si la consulta incluye frases comunes como "Hoja de Vida". Esto se debe a que Azure AI Search puede devolver resultados que contienen esa frase, pero no necesariamente son relevantes para la consulta.

## Solución: Configurar la búsqueda semántica en Azure AI Search

Para mejorar la relevancia de los resultados, podemos configurar la búsqueda semántica en Azure AI Search para que tome en cuenta otras características del índice como el título del documento o palabras clave adicionales.

### Configuración de la búsqueda semántica

Al momento de crear un índice en Azure AI Search podemos crear dos campos para utilizarlos como `title` y `keywords`. Estos campos pueden ser utilizados luego en la configuración de `semanticSearch` para mejorar la relevancia de los resultados.

```python
def create_index_definition(name: str) -> SearchIndex:
    """
    Returns an Azure Cognitive Search index with the given name.
    The index includes a vector search with the default HNSW algorithm
    """
    # The fields we want to index. The "embedding" field is a vector field that will
    # be used for vector search.
    fields=[
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="title", type=SearchFieldDataType.String, searchable=True, filterable=True, facetable=True),
            SearchableField(name="content", type=SearchFieldDataType.String, searchable=True),
            SearchableField(name="keywords", type=SearchFieldDataType.String, searchable=True, filterable=True, facetable=True),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                # Size of the vector created by the text-embedding-3-small model.
                vector_search_dimensions=1536,
                vector_search_profile_name="myHnswProfile",
            ),
        ]

    # The "content" field should be prioritized for semantic ranking.
    semantic_config = SemanticConfiguration(
        name="default",
        prioritized_fields=SemanticPrioritizedFields(
            content_fields=[SemanticField(field_name="content")],
            title_field=SemanticField(field_name="title"),
            keywords_fields=[SemanticField(field_name="keywords")],
        ),
    )

    # For vector search, we want to use the HNSW (Hierarchical Navigable Small World)
    # algorithm (a type of approximate nearest neighbor search algorithm) with cosine
    # distance.
    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="myHnsw")
        ],
        profiles=[
            VectorSearchProfile(
                name="myHnswProfile",
                algorithm_configuration_name="myHnsw",
            )
        ]
    )

    # Create the semantic settings with the configuration
    semantic_search = SemanticSearch(configurations=[semantic_config])

    # Create the search index.
    index = SearchIndex(
        name=name,
        fields=fields,
        semantic_search=semantic_search,
        vector_search=vector_search,
    )

    return index
```

Cuadno indexemos un documento, debemos asegurarnos de incluir los campos `title` y `keywords`. Es estos campos podemos evitar incluir frases comunes y repetitivas como: "Hoja de Vida", "Manual de Usuario", etc. En su lugar, podemos incluir palabras clave que sean más específicas para el contenido del documento.

```python
def gen_index_document() -> List[Dict[str, any]]:
    
    openai_client = openai.OpenAI(
    base_url="https://models.inference.ai.azure.com",
    api_key=os.environ["GITHUB_TOKEN"]
    )
    MODEL_NAME = "text-embedding-3-small"

    content_input = [{
        "Title": "Lionel Messi",
        "Content": "Hoja de vida: Lionel Messi. Futbolista argentino, considerado uno de los mejores jugadores de fútbol de todos los tiempos."
    },
    {
        "Title": "Diego Zumárraga Mera",
        "Content": "Hoja de Vida: Diego Zumárraga Mera. Ingeniero de software con experiencia en desarrollo de aplicaciones web y móviles, apasionado por la inteligencia artificial y el aprendizaje automático."
    }]
    
    items = []
    for ix, item in enumerate(content_input):
        content = item["Content"]
        print(f"Processing item {ix}: {content}")
        embeddings_response = openai_client.embeddings.create(
            model=MODEL_NAME,
            input=content,
        )
        embedding = embeddings_response.data[0].embedding
        print(len(embedding))
        print(embedding)

        items.append({
            "id": f"doc-{ix}",
            "title": item["Title"],
            "content": content,
            "keywords": item["Title"].replace(" ", ", "), # split the title into keywords
            "embedding": embedding
        })

    return items
```
Código completo en el notebook: [AzureAISearchIndex.ipynb](AzureAISearchIndex.ipynb)


### Búsqueda semántica con Azure AI Search

Para probar la búsqueda semántica podemos buscar la frase "Resume la hoja de vida de Diego Zumárraga Mera" y ver cómo Azure AI Search puede devolver resultados relevantes.

```python
query_input = "Resume la hoja de vida de Diego Zumárraga Mera"
embeddings_response = openai_client.embeddings.create(
    model=MODEL_NAME,
    input=query_input,
)
query_embedding = embeddings_response.data[0].embedding
print(query_embedding)

item = {"query": query_input,
        "embedding": query_embedding
}

search_client = SearchClient(
        endpoint=os.environ["AZURE_AI_SEARCH_ENDPOINT"],
        index_name="my-index",
        credential=DefaultAzureCredential(),
    )

results = []
vector_query = VectorizedQuery(
            vector=item["embedding"], k_nearest_neighbors=3, fields="embedding"
        )
result = search_client.search(
    search_text=item["query"],
    vector_queries=[vector_query],
    query_type=QueryType.SEMANTIC,
    semantic_configuration_name="default",
    query_caption=QueryCaptionType.EXTRACTIVE,
    query_answer=QueryAnswerType.EXTRACTIVE,
    top=2,
)
```

Ahora veamos el mejor resultado de la búsqueda:

```python
answers = result.get_answers()
if answers:
    print("Answers:")
    for answer in answers:
        print(f"Answer: {answer.text}")
        print(f"Confidence: {answer.score}")
```
```
Answers:
Answer: Hoja de Vida: Diego Zumárraga Mera. Ingeniero de software con experiencia en desarrollo de aplicaciones web y móviles, apasionado por la inteligencia artificial y el aprendizaje automático.
Confidence: 0.9860000014305115
```
codigo completo en el notebook: [AzureAISearchQuery.ipynb](AzureAISearchQuery.ipynb)

Como podemos ver, Azure AI Search ha devuelto un resultado relevante para la consulta, a pesar de que en la consulta se incluye la frase común **"Hoja de Vida"**. Esto se debe a que hemos configurado la búsqueda semántica para que tenga en cuenta el título y las palabras clave del documento, lo que mejora la relevancia de los resultados.
