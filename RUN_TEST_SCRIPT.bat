REM n times started
for /l %%n in (0, 1, 4) do ( 

	REM  pick database loop (max3)

	for /l %%x in (0, 1, 2) do ( 

		REM pick descriptor loop (max4)

		for /l %%y in (0, 1, 3) do ( 

  			REM  layer size loop (max8)

			for /l %%z in (1, 1, 7) do ( 
   
				REM learning rate loop (max3)
				for /l %%i in (0, 1, 2) do ( 
 
					python NNmodel.py %%x %%y %%z %%i
				)
			)
		)
	)

)
