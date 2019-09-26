REM times started
for /l %%n in (0, 1, 4) do ( 

	REM  database loop max3

	for /l %%x in (0, 1, 2) do ( 

		REM  model shape loop max 5

		for /l %%y in (0, 1, 4) do ( 

			REM  learning rage max 3
			for /l %%i in (0, 1, 2) do ( 

   				echo serie %%n
				python CNNmodel.py %%x %%y %%n %%i
			)
		)
	)
)
