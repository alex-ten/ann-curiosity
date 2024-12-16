using PkgTemplates

template = Template(;
    user = "alex-ten",
    authors = "Alexandr Ten",
    dir = "$(pwd())",
    julia = v"1.10.3"
)

template("IACTools")