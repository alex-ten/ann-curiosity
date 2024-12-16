function Base.show(io::IO, ::MIME"text/plain", l::Letter)
    features = [
        "┃", # 1 
        "▬▬▬▬▬", # 2
        "┃", # 3
        "┃", # 4
        "▬▬▬▬▬", # 5
        "┃", # 6
        "▬▬", # 7
        "┃", # 8
        "▬▬", # 9
        "┃", # 10
        "╲", # 11
        "╱", # 12
        "╲", # 13
        "╱"  # 14
    ]
    v = [replace(f, f[1] => " ") for f in features]
    for k in l.features
        v[k] = features[k]
    end
    d = "$(v[2])\n$(v[1])$(v[11])$(v[8])$(v[12])$(v[3])\n$(v[7]) $(v[9])\n$(v[6])$(v[14])$(v[10])$(v[13])$(v[4])\n$(v[5])\n"
    println(io, d)
end
Base.show(io::IO, l::Letter) = print(io, "Letter($(l.char))")
Base.show(io::IO, w::Word) = print(io, "Word($(w.str))")
function Base.show(io::IO, v::Vocab)
    if size(v.words, 1) < 10
        s = join([w.str for w in v.words], ", ")
    else
        s = join([w.str for w in v.words[1:10]], ", ")
    end
    print(io, "Vocab($s)")
end
Base.size(w::Word) = size(w.letters, 1)
Base.length(w::Word) = length(w.str)
Base.eachindex(w::Word) = eachindex(w.letters)
Base.size(v::Vocab) = size(v.words)
Base.size(v::Vocab, i::Int) = size(v.words, i)

Base.getindex(v::Vocab, i::Int) = v.words[i]
function Base.rand(vocab::Vocab)
    i = rand(1:size(vocab, 1))
    return vocab[i]
end

Base.:(==)(L1::Letter, L2::Letter) = L1.char == L2.char

function Base.rand(font::Vector{Letter}, j::Int, word::Word; ε=.05)
    truth = word.letters[j]
    if rand() < ε
        i = findfirst(==(truth), font)
        return rand(font[InvertedIndex(i)])
    end
    return truth
end

Base.rand(font::Vector{Letter}, word::Word; ε=.05) = [rand(font, j, word; ε=ε) for j in eachindex(word)]

Base.rand(features::Vector{Tuple{Vector{Int64}, Vector{Int64}}}, letter::Letter)