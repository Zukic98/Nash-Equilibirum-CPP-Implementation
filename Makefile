directory = .bin
files = bimatrixEquilibrium osnovniAlgoritam

clean:
	@rm -rf ./$(directory)/

prep_dir:
	@mkdir -p ./$(directory)

compile:
	@$(foreach var, $(files), g++ $(var).cpp -o ./$(directory)/$(var);)

all: clean prep_dir compile
