# ./systemSimulationFromErrorsCSV_no_setup.sh --person P10 --session S3 --window 1 --thresh 100 --actcat Locomotion --sensors backImu --sensors llaImu --sensors luaImu --sensors rtImu --sensors rlaImu --sensors ruaImu

# testPersonsSessions = [('P01', 'S1'), ('P02', 'S2'), ('P03', 'S3'), ('P04', 'S4'), ('P05', 'S1'), ('P06', 'S4')] # final test set
# evaluationPersonsSessions = [('P07', 'S1'), ('P08', 'S4')] # evaluation set
# personsSessions = [('P09', 'S2'), ('P10', 'S3')] # tuning set

# Locomotion no threshold
./systemSimulationFromErrorsCSV_no_setup.sh --person P01 --session S1 --window 15 --thresh 100 --actcat Locomotion
./systemSimulationFromErrorsCSV_no_setup.sh --person P02 --session S2 --window 15 --thresh 100 --actcat Locomotion
./systemSimulationFromErrorsCSV_no_setup.sh --person P03 --session S3 --window 15 --thresh 100 --actcat Locomotion 
./systemSimulationFromErrorsCSV_no_setup.sh --person P04 --session S4 --window 15 --thresh 100 --actcat Locomotion
./systemSimulationFromErrorsCSV_no_setup.sh --person P05 --session S1 --window 15 --thresh 100 --actcat Locomotion
./systemSimulationFromErrorsCSV_no_setup.sh --person P06 --session S4 --window 15 --thresh 100 --actcat Locomotion 
./systemSimulationFromErrorsCSV_no_setup.sh --person P07 --session S1 --window 15 --thresh 100 --actcat Locomotion
./systemSimulationFromErrorsCSV_no_setup.sh --person P08 --session S4 --window 15 --thresh 100 --actcat Locomotion
./systemSimulationFromErrorsCSV_no_setup.sh --person P09 --session S2 --window 15 --thresh 100 --actcat Locomotion 
./systemSimulationFromErrorsCSV_no_setup.sh --person P10 --session S3 --window 15 --thresh 100 --actcat Locomotion