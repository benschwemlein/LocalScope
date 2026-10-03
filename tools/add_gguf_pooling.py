"""
Copy a GGUF embedding model, adding the pooling metadata Ollama needs.

Ollama only treats a GGUF as an embedding model when it declares
<arch>.pooling_type. Some published embedding GGUFs (for example
jina-code-embeddings) omit it, so Ollama loads them as text-completion models
and /api/embeddings fails. This copies the file with that one key added;
every other key and every tensor is copied unchanged.

    python3 tools/add_gguf_pooling.py SRC.gguf DST.gguf POOLING
    printf 'FROM ./DST.gguf\n' > Modelfile && ollama create NAME -f Modelfile

POOLING: 1 = mean, 2 = cls, 3 = last token. Use what the model card says.

No end-of-sequence token is forced on. Check the result against the model's
reference implementation before trusting it: for jina-code-embeddings-0.5b,
appending EOS dropped agreement with the reference to 0.59-0.91 cosine,
while this plain copy agrees at 0.97 or better.

Requires the `gguf` package (pip install gguf).
"""

import sys

import gguf


def main(src: str, dst: str, pooling: int) -> None:
    reader = gguf.GGUFReader(src)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)

    alignment = reader.fields.get(gguf.Keys.General.ALIGNMENT)
    if alignment is not None:
        writer.data_alignment = alignment.contents()

    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
            continue
        vtype = field.types[0]
        sub_type = field.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), vtype, sub_type=sub_type)
    writer.add_uint32(f"{arch}.pooling_type", pooling)

    for tensor in reader.tensors:
        writer.add_tensor_info(tensor.name, tensor.data.shape, tensor.data.dtype,
                               tensor.data.nbytes, tensor.tensor_type)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()
    for tensor in reader.tensors:
        writer.write_tensor_data(tensor.data)
    writer.close()
    print(f"wrote {dst} ({arch}, pooling_type={pooling})")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
