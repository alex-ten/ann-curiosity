using Chain
using CSV
using DataFrames, DataFramesMeta

# Harry Potter (short)
df = @chain CSV.read("iam/def/hp_characters_short.csv", DataFrame; delim=",") begin
    select(Not(:dateOfBirth, :yearOfBirth, :wand, :alive, :image, :actor))
    @transform _ begin
        @byrow begin
            :firstName = ismissing(:name) ? missing : first(split(:name)) |> lowercase
            :lastName = ismissing(:name) ? missing : last(split(:name)) |> lowercase
            :house = ismissing(:house) ? missing : first(:house, 5) |> lowercase
            :ancestry = ismissing(:ancestry) ? missing : first(split(:ancestry, "-"))
            :patronus = ismissing(:patronus) ? missing : replace(:patronus, " " => "_") |> lowercase
            :hogwartsRole = ismissing(:hogwartsStudent) || ismissing(:hogwartsStaff) ? missing : (:hogwartsStudent ? "student" : (:hogwartsStaff ? "staff" : missing))
        end
    end
    select(Not(:hogwartsStudent, :hogwartsStaff, :name))
    rename(:eyeColour => :eyeCol, :hairColour => :hairCol)
    @transform(:id = :firstName .* "_" .* :lastName)
    select([:id, :firstName, :lastName, :species, :gender, :eyeCol, :hairCol, :ancestry, :house, :patronus, :hogwartsRole])
end
CSV.write("iam/def/hp_table.csv", df)