#qca

By Logan Hillberry

This project aims to simulate a quantum versions of the 256 elementary cellular automata (ECA/QECA). 
Enumerate the the possible cellular automata by the rule number R (between 0 and 255). 
Expand R in binary to 8 bits then enumerate the bits by increasing significance. 
The binary expansion of each bit's significance represents the state of a neighborhood and the bit itsel
represents the the next state of the centered neighbor if the neighborhood is indeed in that state. 
I'll call a pairing like this a rule element; each ECA has 8 rule elements. The rule number and its 
corresponding rule elements provides all the information needed to evolve any ECA. For example the 
update table for rule 210 is

              111  110  101  100  011  010  001 000       <-- possible neighborhoods (LSB = 0 to MSB = 7)
               1    1    0    1    0    0    1   0        <-- resulting state of center (210 in binary)

To make a QECA, we first reshape the rule elements by the state of the neighbors (excluding the center site's state)
Thus there are four possible neighbor configurations 11, 10, 01, and 00. For each of these neighbor configurations, 
the center site may take the value 1 or 0. 

                                1 1   1 0   0 1   0 0      <-- Possible neighbors (LSB = 0 to MSB = 3)
                                 ^     ^     ^     ^
                                1 0   1 0   1 0   1 0      <-- possibe center values (either 1 or 0)
                                1 0   1 1   0 1   0 0      <-- resulting state of center (reshaped) 
                              
 Notice that having a 1 0 in the last row implies a state maps to itself, 0 1 maps two states into eachother,
 and 1 1 and 0 0 both map two different states into the same state. Thus, having a 1 1 or 0 0 in the reshaped rule
 implies non-unitary evolution. To handel the non-unitary evolution we will use the operator-sum formalism.
 The basic idea is to create two operator elements for any given rule number. The updating of sites is done
 by allowing each grouping of the reshaped ule number contributes a term to one or both of the operator elements. 
 The "unitary groupings" (i.e. 1 0 and 0 1) conribue to only the first operator element ( I and \sigma_x, respectivly) 
 while the "non-unitary groupings" (i.e 1 1 and 0 0) contribute a term to both operators: 
 0 0 gives |0><0| to the first and |0><1| to the second while 1 1 gives |1><1| to first and |1><1| to second.
 
 There is additional structure to the operator elements which reads the state of a site's neighbors so the correct
 update is applied. This program simulates an ECA by first appending the initial state (a density matrix) to itself,
 and reading the appended copy while updating the original, one site at a time. Note that this severly limits the
 possible size of simulations because we are using a Hilbert space of dim d^2 to simulate the dynamics of our littice
 with dim d. 
 
 It is also possible to simultaniously simulate several ECA rules at once and mix their results at each step. 
 This allows one to think about engineering entanglement dynamics.
