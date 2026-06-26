from embedder import Embedder

embedder = Embedder()

query = "How does approximate nearest neighbor search work?"

v = embedder.encode(query)

print(type(v))
print(len(v))
print(v[0])