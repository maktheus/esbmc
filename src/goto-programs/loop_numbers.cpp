/*******************************************************************\

Module: Loop IDs

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <goto-programs/loop_numbers.h>
#include <util/i2string.h>
#include <util/xml.h>
#include <util/xml_irep.h>

void show_loop_numbers(
  const goto_programt &goto_program)
{
  for(const auto &instruction : goto_program.instructions)
  {
    if(instruction.is_backwards_goto())
    {
      unsigned loop_id = instruction.loop_number;

      
        std::cout << "Loop " << loop_id << ":" << std::endl;

        std::cout << "  " << instruction.location << std::endl;
        std::cout << std::endl;
          }
  }
}

void show_loop_numbers(
  const goto_functionst &goto_functions)
{
  for(const auto &it : goto_functions.function_map)
    show_loop_numbers( it.second.body);
}
