struct Letter
    char::Char
    features::Vector{Int}
    bits::BitVector
    function Letter(c::Char, features::Vector{Int})
        b = BitVector(zeros(14))
        b[features] .= 1
        new(c, features, b)
    end
end

font = [
    Letter('A', [1,2,3,4,6,7,9]),
    Letter('B', [2,3,4,5,8,9,10]),
    Letter('C', [1,2,5,6]),
    Letter('D', [2,3,4,5,8,10]),
    Letter('E', [1,2,5,6,7]),
    Letter('F', [1,2,6,7]),
    Letter('G', [1,2,5,4,5,6,9]),
    Letter('H', [1,3,4,6,7,9]),
    Letter('I', [2,5,8,10]),
    Letter('J', [3,4,5,6]),
    Letter('K', [1,6,7,12,13]),
    Letter('L', [1,5,6]),
    Letter('M', [1,3,4,6,11,12]),
    Letter('N', [1,3,4,6,11,13]),
    Letter('O', [1,2,3,4,5,6]),
    Letter('P', [1,2,3,6,7,9]),
    Letter('Q', [1,2,3,4,5,6,13]),
    Letter('R', [1,2,3,6,7,9,13]),
    Letter('S', [1,2,4,5,7,9]),
    Letter('T', [2,8,10]),
    Letter('U', [1,3,4,5,6]),
    Letter('V', [3,4,11,13]),
    Letter('W', [1,3,4,6,13,14]),
    Letter('X', [11,12,13,14]),
    Letter('Y', [10,11,12]),
    Letter('Z', [2,5,12,14]),
]

struct Word
    letters::Vector{Letter}
    str::String
    function Word(s::String)
        az = collect('A':'Z')
        l = [font[findfirst(==(letter_string), az)] for letter_string in uppercase.(s)]
        new(l, s)
    end
end

struct Vocab
    words::Vector{Word}
    function Vocab(words::Vector{Word})
        new(words)
    end
end

Vocab(words::Vector{String}) = Vocab(Word.(words))


DISPLAY = repeat(repeat(" ", 5) * "\n", 5)

d = """
 ___ 
┃╲┃╱┃
▬▬ ▬▬
┃╱┃╲┃
 ‾‾‾
 """
