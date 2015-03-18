/*******************************************************************\

Module: Program Transformation

Author: Daniel Kroening, kroening@kroening.com
		Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <assert.h>

#include <i2string.h>
#include <cprover_prefix.h>
//#include <expr_util.h>
#include <prefix.h>
#include <std_expr.h>

#include <ansi-c/c_types.h>

#include "goto_convert_class.h"
#include "remove_skip.h"
#include "destructor.h"

#include <arith_tools.h>

//#define DEBUG

#ifdef DEBUG
#define DEBUGLOC std::cout << std::endl << __FUNCTION__ << \
                        "[" << __LINE__ << "]" << std::endl;
#else
#define DEBUGLOC
#endif

/*******************************************************************\

Function: goto_convertt::finish_gotos

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::finish_gotos()
{
  for(gotost::const_iterator it=targets.gotos.begin();
      it!=targets.gotos.end();
      it++)
  {
    goto_programt::instructiont &i=**it;

    if (is_code_goto2t(i.code))
    {
      exprt tmp = migrate_expr_back(i.code);

      const irep_idt &goto_label = tmp.destination();

      labelst::const_iterator l_it = targets.labels.find(goto_label);

      if(l_it==targets.labels.end())
      {
        err_location(tmp);
        std::cerr << "goto label " << goto_label << " not found";
        abort();
      }

      i.targets.clear();
      i.targets.push_back(l_it->second);
    }
    else
    {
      err_location(migrate_expr_back(i.code));
      throw "finish_gotos: unexpected goto";
    }
  }

  targets.gotos.clear();
}

/*******************************************************************\

Function: goto_convertt::goto_convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::goto_convert(const codet &code, goto_programt &dest)
{
  goto_convert_rec(code, dest);
}

/*******************************************************************\

Function: goto_convertt::goto_convert_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::goto_convert_rec(
  const codet &code,
  goto_programt &dest)
{
  convert(code, dest);

  finish_gotos();
}

/*******************************************************************\

Function: goto_convertt::copy

  Inputs:

 Outputs:

 Purpose: Ben: copy code and make a new instruction of goto-functions

\*******************************************************************/

void goto_convertt::copy(
  const codet &code,
  goto_program_instruction_typet type,
  goto_programt &dest)
{
  goto_programt::targett t=dest.add_instruction(type);
  migrate_expr(code, t->code);
  t->location=code.location();
}

/*******************************************************************\

Function: goto_convert::convert_label

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_label(
  const code_labelt &code,
  goto_programt &dest)
{
  if(code.operands().size()!=1)
  {
    err_location(code);
    throw "label statement expected to have one operand";
  }

  // grab the label
  const irep_idt &label=code.get_label();

  goto_programt tmp;

  convert(to_code(code.op0()), tmp);

  // magic ERROR label?

  const std::string &error_label=options.get_option("error-label");

  goto_programt::targett target;

  if(error_label!="" && label==error_label)
  {
    goto_programt::targett t=dest.add_instruction(ASSERT);
    t->guard = false_expr;
    t->location=code.location();
    t->location.property("error label");
    t->location.comment("error label");
    t->location.user_provided(false);

    target=t;
    dest.destructive_append(tmp);
  }
  else
  {
    target=tmp.instructions.begin();
    dest.destructive_append(tmp);
  }

  if(!label.empty())
  {
    targets.labels.insert(std::pair<irep_idt, goto_programt::targett>
                          (label, target));
    target->labels.push_back(label);
  }

  // cases?

  const exprt::operandst &case_op=code.case_op();

  if(!case_op.empty())
  {
    exprt::operandst &case_op_dest=targets.cases[target];

    case_op_dest.reserve(case_op_dest.size()+case_op.size());

    forall_expr(it, case_op)
      case_op_dest.push_back(*it);
  }

  // default?

  if(code.is_default())
    targets.set_default(target);
}

/*******************************************************************\

Function: goto_convertt::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert(
  const codet &code,
  goto_programt &dest)
{
  const irep_idt &statement=code.get_statement();

  if(statement=="block")
    convert_block(code, dest);
  else if(statement=="decl")
    convert_decl(code, dest);
  else if(statement=="expression")
    convert_expression(code, dest);
  else if(statement=="assign")
    convert_assign(to_code_assign(code), dest);
  else if(statement=="init")
    convert_init(code, dest);
  else if(statement=="assert")
    convert_assert(code, dest);
  else if(statement=="assume")
    convert_assume(code, dest);
  else if(statement=="function_call")
    convert_function_call(to_code_function_call(code), dest);
  else if(statement=="label")
    convert_label(to_code_label(code), dest);
  else if(statement=="for")
    convert_for(code, dest);
  else if(statement=="while")
    convert_while(code, dest);
  else if(statement=="dowhile")
    convert_dowhile(code, dest);
  else if(statement=="switch")
    convert_switch(code, dest);
  else if(statement=="break")
    convert_break(to_code_break(code), dest);
  else if(statement=="return")
    convert_return(to_code_return(code), dest);
  else if(statement=="continue")
    convert_continue(to_code_continue(code), dest);
  else if(statement=="goto")
    convert_goto(code, dest);
  else if(statement=="skip")
    convert_skip(code, dest);
  else if(statement=="non-deterministic-goto")
    convert_non_deterministic_goto(code, dest);
  else if(statement=="ifthenelse")
    convert_ifthenelse(code, dest);
  else if(statement=="atomic_begin")
    convert_atomic_begin(code, dest);
  else if(statement=="atomic_end")
    convert_atomic_end(code, dest);
  else if(statement=="cpp_delete" ||
          statement=="cpp_delete[]")
    convert_cpp_delete(code, dest);
  else if(statement=="cpp-catch")
    convert_catch(code, dest);
  else if(statement=="throw_decl")
    convert_throw_decl(code, dest);
  else if(statement=="throw_decl_end")
    convert_throw_decl_end(code,dest);
  else
  {
    copy(code, OTHER, dest);
  }

  // if there is no instruction in the program, add skip to it
  if(dest.instructions.empty())
  {
    dest.add_instruction(SKIP);
    dest.instructions.back().code = expr2tc();
  }
}

/*******************************************************************\

Function: goto_convertt::convert_throw_decl_end

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_throw_decl_end(const exprt &expr, goto_programt &dest)
{
  // add the THROW_DECL_END instruction to 'dest'
  goto_programt::targett throw_decl_end_instruction=dest.add_instruction();
  throw_decl_end_instruction->make_throw_decl_end();
  throw_decl_end_instruction->code = code_cpp_throw_decl_end2tc();
  throw_decl_end_instruction->location=expr.location();
}

/*******************************************************************\

Function: goto_convertt::convert_throw_decl

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_throw_decl(const exprt &expr, goto_programt &dest)
{
  // add the THROW_DECL instruction to 'dest'
  goto_programt::targett throw_decl_instruction=dest.add_instruction();
  codet c("code");
  c.set_statement("throw-decl");

  // the THROW_DECL instruction is annotated with a list of IDs,
  // one per target
  irept::subt &throw_list = c.add("throw_list").get_sub();
  for(unsigned i=0; i<expr.operands().size(); i++)
  {
    const exprt &block=expr.operands()[i];
    irept type = irept(block.get("throw_decl_id"));

    // grab the ID and add to THROW_DECL instruction
    throw_list.push_back(irept(type));
  }

  throw_decl_instruction->make_throw_decl();
  throw_decl_instruction->location=expr.location();
  migrate_expr(c, throw_decl_instruction->code);
}

/*******************************************************************\

Function: goto_convertt::convert_catch

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_catch(
  const codet &code,
  goto_programt &dest)
{
  assert(code.operands().size()>=2);

  // add the CATCH-push instruction to 'dest'
  goto_programt::targett catch_push_instruction=dest.add_instruction();
  catch_push_instruction->make_catch();
  catch_push_instruction->location=code.location();

  // the CATCH-push instruction is annotated with a list of IDs,
  // one per target.
  std::vector<irep_idt> exception_list;

  // add a SKIP target for the end of everything
  goto_programt end;
  goto_programt::targett end_target=end.add_instruction();
  end_target->make_skip();

  // the first operand is the 'try' block
  goto_programt tmp;
  convert(to_code(code.op0()), tmp);
  dest.destructive_append(tmp);

  // add the CATCH-pop to the end of the 'try' block
  goto_programt::targett catch_pop_instruction=dest.add_instruction();
  catch_pop_instruction->make_catch();
  std::vector<irep_idt> empty_excp_list;
  catch_pop_instruction->code = code_cpp_catch2tc(empty_excp_list);

  // add a goto to the end of the 'try' block
  dest.add_instruction()->make_goto(end_target);

  for(unsigned i=1; i<code.operands().size(); i++)
  {
    const codet &block=to_code(code.operands()[i]);

    // grab the ID and add to CATCH instruction
    exception_list.push_back(block.get("exception_id"));

    // Hack for object value passing
    const_cast<exprt::operandst&>(block.op0().operands()).push_back(gen_zero(block.op0().op0().type()));

    convert(block, tmp);
    catch_push_instruction->targets.push_back(tmp.instructions.begin());
    dest.destructive_append(tmp);

    // add a goto to the end of the 'catch' block
    dest.add_instruction()->make_goto(end_target);
  }

  // add end-target
  dest.destructive_append(end);

  catch_push_instruction->code = code_cpp_catch2tc(exception_list);
}

/*******************************************************************\

Function: goto_convertt::convert_block

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_block(
  const codet &code,
  goto_programt &dest)
{
//  if(inductive_step && (const_cast<codet&>(code).add("inside_loop") != irept("")))
//    push_new_loop_block();

  std::list<irep_idt> locals;
  //extract all the local variables from the block

  forall_operands(it, code)
  {
    const codet &code=to_code(*it);

    if(code.get_statement()=="decl")
    {
      const exprt &op0=code.op0();
      assert(op0.id()=="symbol");
      const irep_idt &identifier=op0.identifier();
      const symbolt &symbol=lookup(identifier);

      if(!symbol.static_lifetime &&
         !symbol.type.is_code())
        locals.push_back(identifier);
    }

    goto_programt tmp;
    convert(code, tmp);

    // all the temp symbols are also local variables and they are gotten
    // via the convert process
    for(tmp_symbolst::const_iterator
        it=tmp_symbols.begin();
        it!=tmp_symbols.end();
        it++)
      locals.push_back(*it);

    tmp_symbols.clear();

    //add locals to instructions
    if(!locals.empty())
      Forall_goto_program_instructions(i_it, tmp)
        i_it->add_local_variables(locals);

    dest.destructive_append(tmp);
  }

  // see if we need to call any destructors

  while(!locals.empty())
  {
    const symbolt &symbol=ns.lookup(locals.back());

    code_function_callt destructor=get_destructor(ns, symbol.type);

    if(destructor.is_not_nil())
    {
      // add "this"
      exprt this_expr("address_of", pointer_typet());
      this_expr.type().subtype()=symbol.type;
      this_expr.copy_to_operands(symbol_expr(symbol));
      destructor.arguments().push_back(this_expr);

      goto_programt tmp;
      convert(destructor, tmp);

      Forall_goto_program_instructions(i_it, tmp)
        i_it->add_local_variables(locals);

      dest.destructive_append(tmp);
    }

    locals.pop_back();
  }

//  if(inductive_step && (const_cast<codet&>(code).add("inside_loop") != irept("")))
//  {
//    const_cast<codet&>(code).remove("inside_loop");
//    pop_loop_block();
//  }
}

/*******************************************************************\

Function: goto_convertt::convert_sideeffect

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_sideeffect(
  exprt &expr,
  goto_programt &dest)
{
  const irep_idt &statement=expr.statement();

  if(statement=="postincrement" ||
     statement=="postdecrement" ||
     statement=="preincrement" ||
     statement=="predecrement")
  {
    if(expr.operands().size()!=1)
    {
      err_location(expr);
      str << statement << " takes one argument";
      throw 0;
    }

    if((current_block != NULL) && !is_ifthenelse_block() && inductive_step)
    {
      symbolst::iterator it=context.symbols.find(expr.op0().identifier());
      if(it!=context.symbols.end())
        it->second.value.add("assignment_inside_loop")=irept("1");
    }

    exprt rhs;

    if(statement=="postincrement" ||
       statement=="preincrement")
      rhs.id("+");
    else
      rhs.id("-");

    const typet &op_type=ns.follow(expr.op0().type());

    if(op_type.is_bool())
    {
      rhs.copy_to_operands(expr.op0(), gen_one(int_type()));
      rhs.op0().make_typecast(int_type());
      rhs.type()=int_type();
      rhs.make_typecast(typet("bool"));
    }
    else if(op_type.id()=="c_enum" ||
            op_type.id()=="incomplete_c_enum")
    {
      rhs.copy_to_operands(expr.op0(), gen_one(int_type()));
      rhs.op0().make_typecast(int_type());
      rhs.type()=int_type();
      rhs.make_typecast(op_type);
    }
    else
    {
      typet constant_type;

      if(op_type.id()=="pointer")
        constant_type=index_type();
      else if(is_number(op_type))
        constant_type=op_type;
      else
      {
        err_location(expr);
        throw "no constant one of type "+op_type.to_string();
      }

      exprt constant=gen_one(constant_type);

      rhs.copy_to_operands(expr.op0());
      rhs.move_to_operands(constant);
      rhs.type()=expr.op0().type();
    }

    codet assignment("assign");
    assignment.copy_to_operands(expr.op0());
    assignment.move_to_operands(rhs);

    assignment.location()=expr.find_location();

    convert(assignment, dest);
  }
  else if(statement=="assign")
  {
    exprt tmp;
    tmp.swap(expr);
    tmp.id("code");
    convert(to_code(tmp), dest);
  }
  else if(statement=="assign+" ||
          statement=="assign-" ||
          statement=="assign*" ||
          statement=="assign_div" ||
          statement=="assign_mod" ||
          statement=="assign_shl" ||
          statement=="assign_ashr" ||
          statement=="assign_lshr" ||
          statement=="assign_bitand" ||
          statement=="assign_bitxor" ||
          statement=="assign_bitor")
  {
    if(expr.operands().size()!=2)
    {
      err_location(expr);
      str << statement << " takes two arguments";
      throw 0;
    }

    exprt rhs;

    if(statement=="assign+")
      rhs.id("+");
    else if(statement=="assign-")
      rhs.id("-");
    else if(statement=="assign*")
      rhs.id("*");
    else if(statement=="assign_div")
      rhs.id("/");
    else if(statement=="assign_mod")
      rhs.id("mod");
    else if(statement=="assign_shl")
      rhs.id("shl");
    else if(statement=="assign_ashr")
      rhs.id("ashr");
    else if(statement=="assign_lshr")
      rhs.id("lshr");
    else if(statement=="assign_bitand")
      rhs.id("bitand");
    else if(statement=="assign_bitxor")
      rhs.id("bitxor");
    else if(statement=="assign_bitor")
      rhs.id("bitor");
    else
    {
      err_location(expr);
      str << statement << " not yet supproted";
      throw 0;
    }

    rhs.copy_to_operands(expr.op0(), expr.op1());
    rhs.type()=expr.op0().type();

    if(rhs.op0().type().is_bool())
    {
      rhs.op0().make_typecast(int_type());
      rhs.op1().make_typecast(int_type());
      rhs.type()=int_type();
      rhs.make_typecast(typet("bool"));
    }

    exprt lhs(expr.op0());

    code_assignt assignment(lhs, rhs);
    assignment.location()=expr.location();

    convert(assignment, dest);
  }
  else if(statement=="cpp_delete" ||
          statement=="cpp_delete[]")
  {
    exprt tmp;
    tmp.swap(expr);
    tmp.id("code");
    convert(to_code(tmp), dest);
  }
  else if(statement=="function_call")
  {
    if(expr.operands().size()!=2)
    {
      err_location(expr);
      str << "function_call sideeffect takes two arguments, but got "
          << expr.operands().size();
      throw 0;
    }

    code_function_callt function_call;
    function_call.location()=expr.location();
    function_call.function()=expr.op0();
    function_call.arguments()=expr.op1().operands();
    convert_function_call(function_call, dest);
  }
  else if(statement=="statement_expression")
  {
    if(expr.operands().size()!=1)
    {
      err_location(expr);
      str << "statement_expression sideeffect takes one argument";
      throw 0;
    }

    convert(to_code(expr.op0()), dest);
  }
  else if(statement=="gcc_conditional_expression")
  {
    remove_sideeffects(expr, dest, false);
  }
  else if(statement=="temporary_object")
  {
    remove_sideeffects(expr, dest, false);
  }
  else if(statement=="cpp-throw")
  {
    goto_programt::targett t=dest.add_instruction(THROW);
    codet tmp("cpp-throw");
    tmp.operands().swap(expr.operands());
    tmp.location()=expr.location();
    tmp.set("exception_list", expr.find("exception_list"));
    migrate_expr(tmp, t->code);
    t->location=expr.location();

    // the result can't be used, these are void
    expr.make_nil();
  }
  else
  {
    err_location(expr);
    str << "sideeffect " << statement << " not supported";
    throw 0;
  }
}

/*******************************************************************\

Function: goto_convertt::convert_expression

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_expression(
  const codet &code,
  goto_programt &dest)
{
  if(code.operands().size()!=1)
  {
    err_location(code);
    throw "expression statement takes one operand";
  }

  exprt expr=code.op0();

  if(expr.id()=="sideeffect")
  {
    Forall_operands(it, expr)
      remove_sideeffects(*it, dest);

    goto_programt tmp;
    convert_sideeffect(expr, tmp);
    dest.destructive_append(tmp);
  }
  else
  {
    remove_sideeffects(expr, dest, false); // result not used

    if(expr.is_not_nil())
    {
      codet tmp(code);
      tmp.op0()=expr;
      copy(tmp, OTHER, dest);
    }
  }
}

/*******************************************************************\

Function: goto_convertt::is_expr_in_state

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool goto_convertt::is_expr_in_state(
  const exprt &expr)
{
  if(current_block == NULL)
    return true;

  const struct_typet &struct_type = to_struct_type(current_block->state);
  const struct_typet::componentst &components = struct_type.components();

  for (struct_typet::componentst::const_iterator
     it = components.begin();
     it != components.end();
     it++)
  {
    if (it->get("name").compare(expr.get_string("identifier")) == 0)
   	  return true;
  }

  return false;
}

/*******************************************************************\

Function: goto_convertt::get_struct_components

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::get_struct_components(const exprt &exp, bool is_global)
{
  DEBUGLOC;

  if (exp.is_symbol() && exp.type().id()!="code")
  {
    std::size_t found = exp.identifier().as_string().find("__ESBMC_");
    if(found != std::string::npos)
      return;

    if(exp.identifier().as_string() == "c::__func__"
       || exp.identifier().as_string() == "c::__PRETTY_FUNCTION__"
       || exp.identifier().as_string() == "c::__LINE__"
       || exp.identifier().as_string() == "c::pthread_lib::num_total_threads"
       || exp.identifier().as_string() == "c::pthread_lib::num_threads_running")
      return;

    if(exp.location().file().as_string() == "<built-in>"
       || exp.cmt_location().file().as_string() == "<built-in>"
       || exp.type().location().file().as_string() == "<built-in>"
       || exp.type().cmt_location().file().as_string() == "<built-in>")
      return;

    // That means that we're trying to insert the global variables, but
    // there is no state created here, so add to the global_vars and return
    if (is_global)
    {
      global_vars.insert(exp);
      return;
    }

    if (current_block != NULL)
      current_block->loop_vars.insert(exp);

    if (!is_expr_in_state(exp))
    {
      unsigned int size = current_block->state.components().size();
      current_block->state.components().resize(size+1);
      current_block->state.components()[size] = (struct_typet::componentt &) exp;
      current_block->state.components()[size].set_name(exp.get_string("identifier"));
      current_block->state.components()[size].pretty_name(exp.get_string("identifier"));
    }
  }
  else
  {
    forall_operands(it, exp)
      get_struct_components(*it);
  }
}

/*******************************************************************\

Function: goto_convertt::convert_decl

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_decl(
  const codet &code,
  goto_programt &dest)
{
  if(code.operands().size()!=1 &&
     code.operands().size()!=2)
  {
    err_location(code);
    throw "decl statement takes one or two operands";
  }

  const exprt &op0=code.op0();

  if(op0.id()!="symbol")
  {
    err_location(op0);
    throw "decl statement expects symbol as first operand";
  }

  const irep_idt &identifier=op0.identifier();

  const symbolt &symbol=lookup(identifier);
  if(symbol.static_lifetime ||
     symbol.type.is_code())
	  return; // this is a SKIP!

  if(code.operands().size()==1)
  {
    copy(code, OTHER, dest);
  }
  else
  {
    exprt initializer;
    codet tmp(code);
    initializer=code.op1();
    tmp.operands().resize(1); // just resize the vector, this will get rid of op1

    goto_programt sideeffects;

    if(options.get_bool_option("atomicity-check"))
    {
      unsigned int globals = get_expr_number_globals(initializer);
      if(globals > 0)
        break_globals2assignments(initializer, dest,code.location());
    }

    if(initializer.is_typecast())
    {
      if(initializer.get("cast")=="dynamic")
      {
        exprt op0 = initializer.op0();
        initializer.swap(op0);

        if(!code.op1().is_empty())
        {
          exprt function = code.op1();
          // We must check if the is a exception list
          // If there is, we must throw the exception
          if (function.has_operands())
          {
            if (function.op0().has_operands())
            {
              const exprt& exception_list=
                  static_cast<const exprt&>(function.op0().op0().find("exception_list"));

              if(exception_list.is_not_nil())
              {
                // Let's create an instruction for bad_cast

                // Convert current exception list to a vector of strings.
                std::vector<irep_idt> excp_list;
                forall_irep(it, exception_list.get_sub())
                  excp_list.push_back(it->id());

                // Add new instruction throw
                goto_programt::targett t=dest.add_instruction(THROW);
                t->code = code_cpp_throw2tc(expr2tc(), excp_list);
                t->location=function.location();
              }
            }
          }
          else
          {
            remove_sideeffects(initializer, dest);
            dest.output(std::cout);
          }

          // break up into decl and assignment
          copy(tmp, OTHER, dest);
          code_assignt assign(code.op0(), initializer); // initializer is without sideeffect now
          assign.location()=tmp.location();
          copy(assign, ASSIGN, dest);
          return;
        }
      }
    }
    remove_sideeffects(initializer, sideeffects);
    dest.destructive_append(sideeffects);
    // break up into decl and assignment
    copy(tmp, OTHER, dest);

		if (initializer.is_symbol())
      nondet_vars.insert(std::pair<exprt,exprt>(code.op0(),initializer));

    code_assignt assign(code.op0(), initializer); // initializer is without sideeffect now
    assign.location()=tmp.location();
    copy(assign, ASSIGN, dest);
  }
}

/*******************************************************************\

Function: goto_convertt::convert_assign

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_assign(
  const code_assignt &code,
  goto_programt &dest)
{
  if(code.operands().size()!=2)
  {
    err_location(code);
    throw "assignment statement takes two operands";
  }

  if((current_block != NULL) && !is_ifthenelse_block() && inductive_step)
  {
    symbolst::iterator it=context.symbols.find(code.op0().identifier());
    if(it!=context.symbols.end())
      it->second.value.add("assignment_inside_loop")=irept("1");
  }

  exprt lhs=code.lhs(),
        rhs=code.rhs();

  remove_sideeffects(lhs, dest);

  if(rhs.id()=="sideeffect" &&
     rhs.statement()=="function_call")
  {
    if(rhs.operands().size()!=2)
    {
      err_location(rhs);
      throw "function_call sideeffect takes two operands";
    }

    Forall_operands(it, rhs)
      remove_sideeffects(*it, dest);

    do_function_call(lhs, rhs.op0(), rhs.op1().operands(), dest);
  }
  else if(rhs.id()=="sideeffect" &&
          (rhs.statement()=="cpp_new" ||
           rhs.statement()=="cpp_new[]"))
  {
    Forall_operands(it, rhs)
      remove_sideeffects(*it, dest);

    do_cpp_new(lhs, rhs, dest);
  }
  else
  {
    remove_sideeffects(rhs, dest);

    if (rhs.type().is_code())
    {
      convert(to_code(rhs), dest);
      return;
    }

    if(lhs.id()=="typecast")
    {
      assert(lhs.operands().size()==1);

      // move to rhs
      exprt tmp_rhs(lhs);
      tmp_rhs.op0()=rhs;
      rhs=tmp_rhs;

      // remove from lhs
      exprt tmp(lhs.op0());
      lhs.swap(tmp);
    }

    int atomic = 0;
    if(options.get_bool_option("atomicity-check"))
    {
      unsigned int globals = get_expr_number_globals(lhs);
      atomic = globals;
      globals += get_expr_number_globals(rhs);
      if(globals > 0 && (lhs.identifier().as_string().find("tmp$") == std::string::npos))
        break_globals2assignments(atomic,lhs,rhs, dest,code.location());
    }

    code_assignt new_assign(code);
    new_assign.lhs()=lhs;
    new_assign.rhs()=rhs;
    copy(new_assign, ASSIGN, dest);

    if(options.get_bool_option("atomicity-check"))
      if(atomic == -1)
        dest.add_instruction(ATOMIC_END);
  }

  if (inductive_step && current_block != NULL && lhs.type().id() != "empty")
  {
    get_struct_components(lhs);
  }
}

/*******************************************************************\

Function: goto_convertt::break_globals2assignments

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::break_globals2assignments(int & atomic,exprt &lhs, exprt &rhs, goto_programt &dest, const locationt &location)
{

  if(!options.get_bool_option("atomicity-check"))
    return;

  exprt atomic_dest = exprt("and", typet("bool"));

  /* break statements such as a = b + c as follows:
   * tmp1 = b;
   * tmp2 = c;
   * atomic_begin
   * assert tmp1==b && tmp2==c
   * a = b + c
   * atomic_end
  */
  //break_globals2assignments_rec(lhs,atomic_dest,dest,atomic,location);
  break_globals2assignments_rec(rhs,atomic_dest,dest,atomic,location);

  if(atomic_dest.operands().size()==1)
  {
    exprt tmp;
    tmp.swap(atomic_dest.op0());
    atomic_dest.swap(tmp);
  }
  if(atomic_dest.operands().size() != 0)
  {
	// do an assert
	if(atomic > 0)
	{
	  dest.add_instruction(ATOMIC_BEGIN);
	  atomic = -1;
	}
	goto_programt::targett t=dest.add_instruction(ASSERT);
        expr2tc tmp_guard;
        migrate_expr(atomic_dest, tmp_guard);
        t->guard = tmp_guard;
	t->location=location;
	  t->location.comment("atomicity violation on assignment to " + lhs.identifier().as_string());
  }
}

/*******************************************************************\

Function: goto_convertt::break_globals2assignments

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::break_globals2assignments(exprt & rhs, goto_programt & dest, const locationt & location)
{

  if(!options.get_bool_option("atomicity-check"))
    return;

  if (rhs.operands().size()>0)
    if (rhs.op0().identifier().as_string().find("pthread") != std::string::npos)
 	  return ;

  if (rhs.operands().size()>0)
    if (rhs.op0().operands().size()>0)
 	  return ;

  exprt atomic_dest = exprt("and", typet("bool"));
  break_globals2assignments_rec(rhs,atomic_dest,dest,0,location);

  if(atomic_dest.operands().size()==1)
  {
    exprt tmp;
    tmp.swap(atomic_dest.op0());
    atomic_dest.swap(tmp);
  }

  if(atomic_dest.operands().size() != 0)
  {
    goto_programt::targett t=dest.add_instruction(ASSERT);
    expr2tc tmp_dest;
    migrate_expr(atomic_dest, tmp_dest);
    t->guard.swap(tmp_dest);
    t->location=location;
    t->location.comment("atomicity violation");
  }
}

/*******************************************************************\

Function: goto_convertt::break_globals2assignments_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::break_globals2assignments_rec(exprt &rhs, exprt &atomic_dest, goto_programt &dest, int atomic, const locationt &location)
{

  if(!options.get_bool_option("atomicity-check"))
    return;

  if(rhs.id() == "dereference"
	|| rhs.id() == "implicit_dereference"
	|| rhs.id() == "index"
	|| rhs.id() == "member")
  {
    irep_idt identifier=rhs.op0().identifier();
    if (rhs.id() == "member")
    {
      const exprt &object=rhs.operands()[0];
      identifier=object.identifier();
    }
    else if (rhs.id() == "index")
    {
      identifier=rhs.op1().identifier();
    }

    if (identifier.empty())
	  return;

	const symbolt &symbol=lookup(identifier);

    if (!(identifier == "c::__ESBMC_alloc" || identifier == "c::__ESBMC_alloc_size")
          && (symbol.static_lifetime || symbol.type.is_dynamic_set()))
    {
	  // make new assignment to temp for each global symbol
	  symbolt &new_symbol=new_tmp_symbol(rhs.type());
	  equality_exprt eq_expr;
	  irept irep;
	  new_symbol.to_irep(irep);
	  eq_expr.lhs()=symbol_expr(new_symbol);
	  eq_expr.rhs()=rhs;
	  atomic_dest.copy_to_operands(eq_expr);

	  codet assignment("assign");
	  assignment.reserve_operands(2);
	  assignment.copy_to_operands(symbol_expr(new_symbol));
	  assignment.copy_to_operands(rhs);
	  assignment.location() = location;
	  assignment.comment("atomicity violation");
	  copy(assignment, ASSIGN, dest);

	  if(atomic == 0)
	    rhs=symbol_expr(new_symbol);

    }
  }
  else if(rhs.id() == "symbol")
  {
	const irep_idt &identifier=rhs.identifier();
	const symbolt &symbol=lookup(identifier);
	if(symbol.static_lifetime || symbol.type.is_dynamic_set())
	{
	  // make new assignment to temp for each global symbol
	  symbolt &new_symbol=new_tmp_symbol(rhs.type());
	  new_symbol.static_lifetime=true;
	  equality_exprt eq_expr;
	  irept irep;
	  new_symbol.to_irep(irep);
	  eq_expr.lhs()=symbol_expr(new_symbol);
	  eq_expr.rhs()=rhs;
	  atomic_dest.copy_to_operands(eq_expr);

	  codet assignment("assign");
	  assignment.reserve_operands(2);
	  assignment.copy_to_operands(symbol_expr(new_symbol));
	  assignment.copy_to_operands(rhs);

	  assignment.location() = rhs.find_location();
	  assignment.comment("atomicity violation");
	  copy(assignment, ASSIGN, dest);

	  if(atomic == 0)
	    rhs=symbol_expr(new_symbol);
    }
  }
  else if(!rhs.is_address_of())// && rhs.id() != "dereference")
  {
    Forall_operands(it, rhs)
	{
	  break_globals2assignments_rec(*it,atomic_dest,dest,atomic,location);
	}
  }
}

/*******************************************************************\

Function: goto_convertt::get_expr_number_globals

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

unsigned int goto_convertt::get_expr_number_globals(const exprt &expr)
{
  if(!options.get_bool_option("atomicity-check"))
	return 0;

  if(expr.is_address_of())
  	return 0;

  else if(expr.id() == "symbol")
  {
    const irep_idt &identifier=expr.identifier();
  	const symbolt &symbol=lookup(identifier);

    if (identifier == "c::__ESBMC_alloc"
    	|| identifier == "c::__ESBMC_alloc_size")
    {
      return 0;
    }
    else if (symbol.static_lifetime || symbol.type.is_dynamic_set())
    {
      return 1;
    }
  	else
  	{
  	  return 0;
  	}
  }

  unsigned int globals = 0;

  forall_operands(it, expr)
    globals += get_expr_number_globals(*it);

  return globals;
}

unsigned int goto_convertt::get_expr_number_globals(const expr2tc &expr)
{
  if (is_nil_expr(expr))
    return 0;

  if (!options.get_bool_option("atomicity-check"))
	return 0;

  if (is_address_of2t(expr))
  	return 0;
  else if (is_symbol2t(expr))
  {
    irep_idt identifier = to_symbol2t(expr).get_symbol_name();
    const symbolt &symbol = lookup(identifier);

    if (identifier == "c::__ESBMC_alloc"
    	|| identifier == "c::__ESBMC_alloc_size")
    {
      return 0;
    }
    else if (symbol.static_lifetime || symbol.type.is_dynamic_set())
    {
      return 1;
    }
  	else
  	{
  	  return 0;
  	}
  }

  unsigned int globals = 0;

  forall_operands2(it, idx, expr)
    globals += get_expr_number_globals(*it);

  return globals;
}



/*******************************************************************\

Function: goto_convertt::convert_init

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_init(
  const codet &code,
  goto_programt &dest)
{
  if(code.operands().size()!=2)
  {
    err_location(code);
    throw "init statement takes two operands";
  }

  // make it an assignment
  codet assignment=code;
  assignment.set_statement("assign");

  convert(to_code_assign(assignment), dest);
}

/*******************************************************************\

Function: goto_convertt::convert_cpp_delete

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_cpp_delete(
  const codet &code,
  goto_programt &dest)
{
  if(code.operands().size()!=1)
  {
    err_location(code);
    throw "cpp_delete statement takes one operand";
  }

  exprt tmp_op=code.op0();

  // we call the destructor, and then free
  const exprt &destructor=
    static_cast<const exprt &>(code.find("destructor"));

  if(destructor.is_not_nil())
  {
    if(code.statement()=="cpp_delete[]")
    {
      // build loop
    }
    else if(code.statement()=="cpp_delete")
    {
      exprt deref_op("dereference", tmp_op.type().subtype());
      deref_op.copy_to_operands(tmp_op);

      codet tmp_code=to_code(destructor);
      replace_new_object(deref_op, tmp_code);
      convert(tmp_code, dest);
    }
    else
      assert(0);
  }

  expr2tc tmp_op2;
  migrate_expr(tmp_op, tmp_op2);

  // preserve the call
  goto_programt::targett t_f=dest.add_instruction(OTHER);
  t_f->code = code_cpp_delete2tc(tmp_op2);
  t_f->location=code.location();

  // now do "delete"
  exprt valid_expr("valid_object", bool_typet());
  valid_expr.copy_to_operands(tmp_op);

  // clear alloc bit
  exprt assign = code_assignt(valid_expr, false_exprt());
  expr2tc assign2;
  migrate_expr(assign, assign2);
  goto_programt::targett t_c=dest.add_instruction(ASSIGN);
  t_c->code = assign2;
  t_c->location=code.location();

  exprt deallocated_expr("deallocated_object", bool_typet());
  deallocated_expr.copy_to_operands(tmp_op);

  //indicate that memory has been deallocated
  assign = code_assignt(deallocated_expr, true_exprt());
  migrate_expr(assign, assign2);
  goto_programt::targett t_d=dest.add_instruction(ASSIGN);
  t_d->code = assign2;
  t_d->location=code.location();
}

/*******************************************************************\

Function: goto_convertt::convert_assert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_assert(
  const codet &code,
  goto_programt &dest)
{

  if(code.operands().size()!=1)
  {
    err_location(code);
    throw "assert statement takes one operand";
  }

  exprt cond=code.op0();

  remove_sideeffects(cond, dest);

  if(options.get_bool_option("no-assertions"))
    return;

  if(options.get_bool_option("atomicity-check"))
  {
    unsigned int globals = get_expr_number_globals(cond);
    if(globals > 0)
	  break_globals2assignments(cond, dest,code.location());
  }

  goto_programt::targett t=dest.add_instruction(ASSERT);
  expr2tc tmp_cond;
  migrate_expr(cond, tmp_cond);
  t->guard = tmp_cond;
  t->location=code.location();
  t->location.property("assertion");
  t->location.user_provided(true);
}

/*******************************************************************\

Function: goto_convertt::convert_skip

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_skip(
  const codet &code,
  goto_programt &dest)
{
  goto_programt::targett t=dest.add_instruction(SKIP);
  t->location=code.location();
  expr2tc tmp_code;
  migrate_expr(code, tmp_code);
  t->code = tmp_code;;
}

/*******************************************************************\

Function: goto_convertt::convert_assert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_assume(
  const codet &code,
  goto_programt &dest)
{
  if(code.operands().size()!=1)
  {
    err_location(code);
    throw "assume statement takes one operand";
  }

  exprt op=code.op0();

  remove_sideeffects(op, dest);

  if(options.get_bool_option("atomicity-check"))
  {
    unsigned int globals = get_expr_number_globals(op);
    if(globals > 0)
	  break_globals2assignments(op, dest,code.location());
  }

  goto_programt::targett t=dest.add_instruction(ASSUME);
  expr2tc tmp_op;
  migrate_expr(op, tmp_op);
  t->guard.swap(tmp_op);
  t->location=code.location();
}

/*******************************************************************\

Function: goto_convertt::update_state_vector

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::update_state_vector(
  array_typet state_vector,
  goto_programt &dest)
{
  //set the type of the state vector
  state_vector.subtype() = current_block->state;

  std::string identifier;
  identifier = "kindice$"+i2string(current_block->state_counter);

  exprt lhs_index = symbol_exprt(identifier, int_type());
  exprt new_expr(exprt::with, state_vector);
  exprt lhs_array("symbol", state_vector);
  exprt rhs("symbol", current_block->state);

  std::string identifier_lhs, identifier_rhs;
  identifier_lhs = "s$"+i2string(current_block->state_counter);
  identifier_rhs = "cs$"+i2string(current_block->state_counter);

  lhs_array.identifier(identifier_lhs);
  rhs.identifier(identifier_rhs);

  //s[k]=cs
  new_expr.reserve_operands(3);
  new_expr.copy_to_operands(lhs_array);
  new_expr.copy_to_operands(lhs_index);
  new_expr.move_to_operands(rhs);
  code_assignt new_assign(lhs_array,new_expr);
  copy(new_assign, ASSIGN, dest);
}

/*******************************************************************\

Function: goto_convertt::assume_all_state_vector

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::assume_all_state_vector(
  array_typet state_vector,
  goto_programt &dest)
{
  // Temp symbol that will be used to count up to kindice
  exprt tmp_symbol = gen_zero(int_type());
  make_temp_symbol(tmp_symbol, dest);

  // Condition (tmp_symbol <= kindice)
  exprt lhs_index =
    symbol_exprt("kindice$"+i2string(current_block->state_counter), int_type());
  exprt cond("<=", typet("bool"));
  cond.copy_to_operands(tmp_symbol, lhs_index);

  // save break/continue targets
  break_continue_targetst old_targets(targets);

  // do the v label
  goto_programt tmp_v;
  goto_programt::targett v=tmp_v.add_instruction();

  // do the z label
  goto_programt tmp_z;
  goto_programt::targett z=tmp_z.add_instruction(SKIP);

  // Increment (tmp_symbol++)
  side_effect_exprt postincrement("postincrement");
  postincrement.type() = int_type();
  postincrement.copy_to_operands(tmp_symbol);

  code_expressiont increment_expression;
  increment_expression.copy_to_operands(postincrement);

  // do the x label
  goto_programt tmp_x;
  convert(to_code(increment_expression), tmp_x);

  // do the u label
    goto_programt::targett u=v;

  // set the targets
  targets.set_break(z);
  targets.set_continue(tmp_x.instructions.begin());

  // v: if(!c) goto z;
  v->make_goto(z);
  expr2tc tmp_cond;
  migrate_expr(cond, tmp_cond);
  tmp_cond = not2tc(tmp_cond);
  v->guard = tmp_cond;

  // do the w label
  goto_programt tmp_w;
  //set the type of the state vector
  state_vector.subtype() = current_block->state;

  exprt new_expr(exprt::index, current_block->state);
  exprt lhs_array("symbol", state_vector);
  exprt rhs("symbol", current_block->state);

  lhs_array.identifier("s$"+i2string(current_block->state_counter));
  rhs.identifier("cs$"+i2string(current_block->state_counter));

  //s[k]
  new_expr.reserve_operands(2);
  new_expr.copy_to_operands(lhs_array);
  new_expr.copy_to_operands(tmp_symbol);

  //assume(s[k]!=cs)
  exprt result_expr = gen_binary(exprt::notequal, bool_typet(), new_expr, rhs);
  assume_cond(result_expr, false, tmp_w);

  // y: goto u;
  goto_programt tmp_y;
  goto_programt::targett y=tmp_y.add_instruction();
  y->make_goto(u);
  y->guard = true_expr;

  dest.destructive_append(tmp_v);
  dest.destructive_append(tmp_w);
  dest.destructive_append(tmp_x);
  dest.destructive_append(tmp_y);
  dest.destructive_append(tmp_z);

  // restore break/continue
  targets.restore(old_targets);

  //kindice=kindice+1
  exprt one_expr = gen_one(int_type());
  exprt rhs_expr = gen_binary(exprt::plus, int_type(), lhs_index, one_expr);
  code_assignt new_assign_plus(lhs_index,rhs_expr);
  copy(new_assign_plus, ASSIGN, dest);
}

/*******************************************************************\

Function: goto_convertt::assume_state_vector

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::assume_state_vector(
  array_typet state_vector,
  goto_programt &dest)
{
  if(assume_all_states)
  {
    assume_all_state_vector(state_vector, dest);
    return;
  }

  //set the type of the state vector
  state_vector.subtype() = current_block->state;

  std::string identifier;
  identifier = "kindice$"+i2string(current_block->state_counter);

  exprt lhs_index = symbol_exprt(identifier, int_type());
  exprt new_expr(exprt::index, current_block->state);
  exprt lhs_array("symbol", state_vector);
  exprt rhs("symbol", current_block->state);

  std::string identifier_lhs, identifier_rhs;

  identifier_lhs = "s$"+i2string(current_block->state_counter);
  identifier_rhs = "cs$"+i2string(current_block->state_counter);

  lhs_array.identifier(identifier_lhs);
  rhs.identifier(identifier_rhs);

  //s[k]
  new_expr.reserve_operands(2);
  new_expr.copy_to_operands(lhs_array);
  new_expr.copy_to_operands(lhs_index);

  //assume(s[k]!=cs)
  exprt result_expr = gen_binary(exprt::notequal, bool_typet(), new_expr, rhs);
  assume_cond(result_expr, false, dest);

  //kindice=kindice+1
  exprt one_expr = gen_one(int_type());
  exprt rhs_expr = gen_binary(exprt::plus, int_type(), lhs_index, one_expr);
  code_assignt new_assign_plus(lhs_index,rhs_expr);
  copy(new_assign_plus, ASSIGN, dest);
}

/*******************************************************************\

Function: goto_convertt::convert_for

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_for(
  const codet &code,
  goto_programt &dest)
{
  if (inductive_step || base_case)
    push_new_loop_block();

  if(code.operands().size()!=4)
  {
    err_location(code);
    throw "for takes four operands";
  }

  // turn for(A; c; B) { P } into
  //  A; while(c) { P; B; }
  //-----------------------------
  //    A;
  // t: cs=nondet
  // u: sideeffects in c
  // c: k=0
  // v: if(!c) goto z;
  // f: s[k]=cs
  // w: P;
  // x: B;               <-- continue target
  // d: cs assign
  // e: assume
  // y: goto u;
  // z: ;                <-- break target
  // g: assume(!c)

  // A;
  code_blockt block;
  if(code.op0().is_not_nil())
  {
    block.copy_to_operands(code.op0());
    convert(block, dest);
  }

  exprt tmp=code.op1();

  if (inductive_step)
  {
    exprt tmp_op0 = code.op0();
    check_loop_cond(tmp_op0);
    check_loop_cond(tmp);
  }

  exprt cond=tmp;

  array_typet state_vector;

  // do the t label
  if (is_inductive_step_active())
    get_struct_components(code.op3());

  goto_programt sideeffects;

  remove_sideeffects(cond, sideeffects);

  // save break/continue targets
  break_continue_targetst old_targets(targets);

  // do the u label
  goto_programt::targett u=sideeffects.instructions.begin();

  // do the v label
  goto_programt tmp_v;
  goto_programt::targett v=tmp_v.add_instruction();

  // do the z label
  goto_programt tmp_z;
  goto_programt::targett z=tmp_z.add_instruction(SKIP);

  // do the x label
  goto_programt tmp_x;
  if(code.op2().is_nil())
    tmp_x.add_instruction(SKIP);
  else
  {
    exprt tmp_B=code.op2();
    convert(to_code(code.op2()), tmp_x);
  }

  // optimize the v label
  if(sideeffects.instructions.empty())
    u=v;

  // set the targets
  targets.set_break(z);
  targets.set_continue(tmp_x.instructions.begin());

  // v: if(!c) goto z;
  v->make_goto(z);
  expr2tc tmp_cond;
  migrate_expr(cond, tmp_cond);
  tmp_cond = not2tc(tmp_cond);
  v->guard = tmp_cond;
  v->location=cond.location();

  // do the w label
  goto_programt tmp_w;
  convert(to_code(code.op3()), tmp_w);

  if (is_inductive_step_active())
  {
    make_nondet_assign(dest);
    init_k_indice(dest);
  }

  // y: goto u;
  goto_programt tmp_y;
  goto_programt::targett y=tmp_y.add_instruction();
  y->make_goto(u);
  y->guard = true_expr;
  y->location=code.location();

  dest.destructive_append(sideeffects);
  dest.destructive_append(tmp_v);

  // do the f label
  if (is_inductive_step_active())
    update_state_vector(state_vector, dest);

  dest.destructive_append(tmp_w);
  dest.destructive_append(tmp_x);

  // do the d label
  if (is_inductive_step_active())
    assign_current_state(dest);

  // do the e label
  if (is_inductive_step_active())
    assume_state_vector(state_vector, dest);

  dest.destructive_append(tmp_y);
  dest.destructive_append(tmp_z);

  //do the g label
  if (inductive_step || base_case)
    assume_cond(cond, true, dest); //assume(!c)
  else if (forward_condition)
    assert_cond(cond, true, dest); //assert(!c)

  // restore break/continue
  targets.restore(old_targets);

  if (inductive_step || base_case)
    pop_loop_block();
}

/*******************************************************************\

Function: goto_convertt::add_global_variable_to_state

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::add_global_variable_to_state()
{
  if(!inductive_step)
    return;

  forall_symbols(it, context.symbols) {
    if(it->second.static_lifetime && !it->second.type.is_pointer())
    {
      exprt s = symbol_expr(it->second);
      if(it->second.value.id()==irep_idt("array_of"))
        s.type()=it->second.value.type();
      get_struct_components(s, true);
    }
  }
}

/*******************************************************************\

Function: goto_convertt::make_nondet_assign

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::make_nondet_assign(
  goto_programt &dest)
{
  for (unsigned int j=0; j < current_block->state.components().size(); j++)
  {
    exprt rhs_expr=side_effect_expr_nondett(current_block->state.components()[j].type());
    exprt new_expr(exprt::with, current_block->state);
    exprt lhs_expr("symbol", current_block->state);

    if (current_block->state.components()[j].type().is_array())
      rhs_expr = side_effect_expr_nondett(current_block->state.components()[j].type());

    std::string identifier;
    identifier = "cs$"+i2string(current_block->state_counter);
    lhs_expr.identifier(identifier);

    new_expr.reserve_operands(3);
    new_expr.copy_to_operands(lhs_expr);
    new_expr.copy_to_operands(exprt("member_name"));
    new_expr.move_to_operands(rhs_expr);

    if (!current_block->state.components()[j].operands().size())
    {
      new_expr.op1().component_name(current_block->state.components()[j].get_string("identifier"));
      assert(!new_expr.op1().get_string("component_name").empty());
    }
    else
    {
      forall_operands(it, current_block->state.components()[j])
      {
        new_expr.op1().component_name(it->get_string("identifier"));
        assert(!new_expr.op1().get_string("component_name").empty());
      }
    }
    code_assignt new_assign(lhs_expr,new_expr);
    copy(new_assign, ASSIGN, dest);
  }
}

/*******************************************************************\

Function: goto_convertt::assign_current_state

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::assign_current_state(
  goto_programt &dest)
{
  for (unsigned int j = 0; j < current_block->state.components().size(); j++)
  {
    exprt rhs_expr(current_block->state.components()[j]);
    exprt new_expr(exprt::with, current_block->state);
    exprt lhs_expr("symbol", current_block->state);

    std::string identifier;

    identifier = "cs$" + i2string(current_block->state_counter);

    lhs_expr.identifier(identifier);

    new_expr.reserve_operands(3);
    new_expr.copy_to_operands(lhs_expr);
    new_expr.copy_to_operands(exprt("member_name"));
    new_expr.move_to_operands(rhs_expr);

    if (!current_block->state.components()[j].operands().size())
    {
      new_expr.op1().component_name(
        current_block->state.components()[j].get_string("identifier"));
      assert(!new_expr.op1().get_string("component_name").empty());
    }
    else
    {
      forall_operands(it, current_block->state.components()[j])
      {
        new_expr.op1().component_name(it->get_string("identifier"));
        assert(!new_expr.op1().get_string("component_name").empty());
      }
    }

    code_assignt new_assign(lhs_expr, new_expr);
    copy(new_assign, ASSIGN, dest);
  }
}

/*******************************************************************\

Function: goto_convertt::init_k_indice

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::init_k_indice(
  goto_programt &dest)
{
  std::string identifier;
  identifier = "kindice$"+i2string(current_block->state_counter);
  exprt lhs_index = symbol_exprt(identifier, int_type());
  exprt zero_expr = gen_zero(int_type());
  code_assignt new_assign(lhs_index,zero_expr);
  copy(new_assign, ASSIGN, dest);
}

/*******************************************************************\

Function: goto_convertt::assume_cond

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::assume_cond(
  const exprt &cond,
  const bool &neg,
  goto_programt &dest)
{
  // We don't assume false if there is a break inside the loop
  if(cond.is_true() && current_block->has_break)
    return;

  goto_programt tmp_e;
  goto_programt::targett e=tmp_e.add_instruction(ASSUME);
  exprt result_expr = cond;
  if (neg)
    result_expr.make_not();

  remove_sideeffects(result_expr, dest, true);

  migrate_expr(result_expr, e->guard);

  dest.destructive_append(tmp_e);
}

/*******************************************************************\

Function: goto_convertt::assert_cond

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::assert_cond(
  const exprt &cond,
  const bool &neg,
  goto_programt &dest)
{
  // We don't assert false if there is a break inside the loop
  if(cond.is_true() && current_block->has_break)
    return;

  goto_programt tmp_e;
  goto_programt::targett e=tmp_e.add_instruction(ASSERT);
  exprt result_expr = cond;
  if (neg)
    result_expr.make_not();

  migrate_expr(result_expr, e->guard);

  dest.destructive_append(tmp_e);
}

/*******************************************************************\

Function: goto_convertt::disable_k_induction

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::disable_k_induction()
{
  options.set_option("disable-inductive-step", true);
  inductive_step=false;
  base_case=false;
}


/*******************************************************************\

Function: goto_convertt::print_msg_mem_alloc

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::print_msg_mem_alloc(void)
{
  std::cerr << "warning: this program contains dynamic memory allocation,"
            << " so we are not applying the inductive step to this program!"
            << std::endl;
  std::cout << "failed" << std::endl;
  disable_k_induction();
}

/*******************************************************************\

Function: goto_convertt::print_msg

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::print_msg(
  const exprt &tmp)
{
  std::cerr << "warning: this program " << tmp.location().get_file()
            << " contains a '" << tmp.id() << "' operator at line "
            << tmp.location().get_line()
            << ", so we are not applying the inductive step to this program!"
            << std::endl;
  disable_k_induction();
}

/*******************************************************************\

Function: goto_convertt::check_op_const

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool goto_convertt::check_expr_const(
  const exprt &expr,
  const locationt &loc)
{
  if (expr.is_constant() || expr.type().id() == "pointer")
    return true;

  if (expr.is_symbol())
    if(ns.lookup(expr.identifier()).value.is_constant())
      return true;

  return false;
}

/*******************************************************************\

Function: goto_convertt::init_nondet_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::init_nondet_expr(
  exprt &tmp,
  goto_programt &dest)
{
  if (!tmp.is_symbol())
    return;

  exprt nondet_expr = side_effect_expr_nondett(tmp.type());
  code_assignt new_assign_nondet(tmp, nondet_expr);
  copy(new_assign_nondet, ASSIGN, dest);

  if (!is_ifthenelse_block())
    nondet_vars.insert(std::pair<exprt, exprt>(tmp, nondet_expr));
}

/*******************************************************************\

Function: goto_convertt::set_expr_to_nondet

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::set_expr_to_nondet(
  exprt &tmp,
  goto_programt &dest)
{
  nondet_varst::const_iterator cache_result;
  if (tmp.op0().is_constant())
  {
    cache_result = nondet_vars.find(tmp.op1());
    if (cache_result == nondet_vars.end())
      init_nondet_expr(tmp.op1(), dest);
  }
  else
  {
    cache_result = nondet_vars.find(tmp.op0());
    if (cache_result == nondet_vars.end())
      init_nondet_expr(tmp.op0(), dest);
  }
}

/*******************************************************************\

Function: goto_convertt::check_loop_cond

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::check_loop_cond(
  exprt &expr)
{
  assert(current_block != NULL);

  irep_idt exprid = expr.id();

  if (expr.is_true())
  {
    // allow transformations on infinite loops
    current_block->active = true;
  }
  else if (expr.is_false() || expr.is_constant())
  {

  }
  else if (exprid == ">" ||  exprid == ">=" || expr.is_typecast())
  {
    if (!check_expr_const(expr.op0(), expr.location()))
      current_block->active = true;
  }
  else if (exprid == "<" ||  exprid == "<=")
  {
    if (!check_expr_const(expr.op1(), expr.location()))
      current_block->active = true;
  }
  else if (expr.is_and() || expr.is_or()
    || exprid == "-" || exprid == "+"
    || exprid == "*" || exprid == "/"
    || expr.is_notequal())
  {
    assert(expr.operands().size()==2);

    check_loop_cond(expr.op0());
    check_loop_cond(expr.op1());
  }
  else if (expr.is_dereference())
  {
    check_loop_cond(expr.op0());
  }
  else if (expr.is_symbol())
  {
    if (!check_expr_const(expr, expr.location()))
      current_block->active = true;
  }
  else if (expr.is_code())
  {
    if(expr.statement() == "assign")
      check_loop_cond(expr.op1());
  }
  else
  {
    std::cerr << "warning: the expression '" << expr.id()
	      << "' located at line " << expr.location().get_line()
	      << " of " << expr.location().get_file()
  	      << " is not supported yet" << std::endl;
    disable_k_induction();
  }
}

/*******************************************************************\

Function: goto_convertt::convert_while

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_while(
  const codet &code,
  goto_programt &dest)
{
  if (inductive_step || base_case)
    push_new_loop_block();

  if(code.operands().size()!=2)
  {
    err_location(code);
    throw "while takes two operands";
  }

  exprt tmp=code.op0();

  if (inductive_step)
    check_loop_cond(tmp);

  array_typet state_vector;
  const exprt *cond=&tmp;
  const locationt &location=code.location();

  //    while(c) P;
  //--------------------
  // t: cs=nondet
  // v: sideeffects in c
  // c: k=0
  // v: if(!c) goto z;
  // f: s[k]=cs
  // x: P;
  // d: cs assign
  // e: assume
  // y: goto v;          <-- continue target
  // z: ;                <-- break target
  // g: assume(!c)

  // do the t label
  if (is_inductive_step_active())
    get_struct_components(code.op1());

  // save break/continue targets
  break_continue_targetst old_targets(targets);

  // do the z label
  goto_programt tmp_z;
  goto_programt::targett z=tmp_z.add_instruction();
  z->make_skip();

  goto_programt tmp_branch;
  generate_conditional_branch(gen_not(*cond), z, location, tmp_branch);

  // do the v label
  goto_programt::targett v=tmp_branch.instructions.begin();

  // do the y label
  goto_programt tmp_y;
  goto_programt::targett y=tmp_y.add_instruction();

  // set the targets
  targets.set_break(z);
  targets.set_continue(y);

  // do the x label
  goto_programt tmp_x;

  convert(to_code(code.op1()), tmp_x);

  // y: if(c) goto v;
  y->make_goto(v);
  y->guard = true_expr;
  y->location=code.location();

  //do the c label
  if (is_inductive_step_active())
  {
    make_nondet_assign(dest);
    init_k_indice(dest);
  }

  dest.destructive_append(tmp_branch);

  // do the f label
  if (is_inductive_step_active())
    update_state_vector(state_vector, dest);

  dest.destructive_append(tmp_x);

  // do the d label
  if (is_inductive_step_active())
    assign_current_state(dest);

  // do the e label
  if (is_inductive_step_active())
    assume_state_vector(state_vector, dest);

  dest.destructive_append(tmp_y);

  dest.destructive_append(tmp_z);

  //do the g label
  if (inductive_step || base_case)
    assume_cond(*cond, true, dest); //assume(!c)
  else if (forward_condition)
    assert_cond(tmp, true, dest); //assert(!c)

  // restore break/continue
  targets.restore(old_targets);

  if (inductive_step || base_case)
    pop_loop_block();
}

/*******************************************************************\

Function: goto_convertt::convert_dowhile

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_dowhile(
  const codet &code,
  goto_programt &dest)
{
  if (inductive_step || base_case)
    push_new_loop_block();

  if(code.operands().size()!=2)
  {
    err_location(code);
    throw "dowhile takes two operands";
  }

  // save location
  locationt condition_location=code.op0().find_location();

  exprt tmp=code.op0();

  goto_programt sideeffects;
  remove_sideeffects(tmp, sideeffects);

  if (inductive_step)
    check_loop_cond(tmp);

  array_typet state_vector;
  const exprt &cond=tmp;

  //    do P while(c);
  //--------------------
  // t: cs=nondet
  // w: P;
  // x: sideeffects in c   <-- continue target
  // c: k=0
  // y: if(c) goto w;
  // f: s[k]=cs
  // d: cs assign
  // e: assume
  // z: ;                  <-- break target
  // g: assume(!c)

  // do the t label
  if (is_inductive_step_active())
    get_struct_components(code.op1());

  // save break/continue targets
  break_continue_targetst old_targets(targets);

  // do the y label
  goto_programt tmp_y;
  goto_programt::targett y=tmp_y.add_instruction();

  // do the z label
  goto_programt tmp_z;
  goto_programt::targett z=tmp_z.add_instruction();
  z->make_skip();

  // do the x label
  goto_programt::targett x;
  if(sideeffects.instructions.empty())
    x=y;
  else
    x=sideeffects.instructions.begin();

  // set the targets
  targets.set_break(z);
  targets.set_continue(x);

  // do the w label
  goto_programt tmp_w;
  convert(to_code(code.op1()), tmp_w);
  goto_programt::targett w=tmp_w.instructions.begin();

  //do the c label
  if (is_inductive_step_active())
  {
    init_k_indice(dest);
    make_nondet_assign(dest);
  }

  dest.destructive_append(tmp_w);
  dest.destructive_append(sideeffects);

#if 0
  if(options.get_bool_option("atomicity-check"))
  {
    unsigned int globals = get_expr_number_globals(cond);
    if(globals > 0)
	  break_globals2assignments(cond, dest,code.location());
  }
#endif

  // y: if(c) goto w;
  y->make_goto(w);

  migrate_expr(cond, y->guard);

  y->location=condition_location;

  dest.destructive_append(tmp_y);

  // do the f label
  if (is_inductive_step_active())
    update_state_vector(state_vector, dest);

  // do the d label
  if (is_inductive_step_active())
    assign_current_state(dest);

  // do the e label
  if (is_inductive_step_active())
    assume_state_vector(state_vector, dest);

  dest.destructive_append(tmp_z);

  //do the g label
  if (inductive_step || base_case)
    assume_cond(cond, true, dest); //assume(!c)
  else if (forward_condition)
    assert_cond(tmp, true, dest); //assert(!c)

  // restore break/continue targets
  targets.restore(old_targets);

  if (inductive_step || base_case)
    pop_loop_block();
}

/*******************************************************************\

Function: goto_convertt::case_guard

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::case_guard(
  const exprt &value,
  const exprt::operandst &case_op,
  exprt &dest)
{
  dest=exprt("or", typet("bool"));
  dest.reserve_operands(case_op.size());

  forall_expr(it, case_op)
  {
    equality_exprt eq_expr;
    eq_expr.lhs()=value;
    eq_expr.rhs()=*it;
    dest.move_to_operands(eq_expr);
  }

  assert(dest.operands().size()!=0);

  if(dest.operands().size()==1)
  {
    exprt tmp;
    tmp.swap(dest.op0());
    dest.swap(tmp);
  }
}

/*******************************************************************\

Function: goto_convertt::convert_switch

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_switch(
  const codet &code,
  goto_programt &dest)
{
  // switch(v) {
  //   case x: Px;
  //   case y: Py;
  //   ...
  //   default: Pd;
  // }
  // --------------------
  // x: if(v==x) goto X;
  // y: if(v==y) goto Y;
  //    goto d;
  // X: Px;
  // Y: Py;
  // d: Pd;
  // z: ;

  if(code.operands().size()<2)
  {
    err_location(code);
    throw "switch takes at least two operands";
  }

  exprt argument=code.op0();

  goto_programt sideeffects;
  remove_sideeffects(argument, sideeffects);

  // save break/continue/default/cases targets
  break_continue_switch_targetst old_targets(targets);

  // do the z label
  goto_programt tmp_z;
  goto_programt::targett z=tmp_z.add_instruction();
  z->make_skip();

  // set the new targets -- continue stays as is
  targets.set_break(z);
  targets.set_default(z);
  targets.cases.clear();

  goto_programt tmp;

  forall_operands(it, code)
    if(it!=code.operands().begin())
    {
      goto_programt t;
      convert(to_code(*it), t);
      tmp.destructive_append(t);
    }

  goto_programt tmp_cases;

  for(casest::iterator it=targets.cases.begin();
      it!=targets.cases.end();
      it++)
  {
    const caset &case_ops=it->second;

    assert(!case_ops.empty());

    exprt guard_expr;
    case_guard(argument, case_ops, guard_expr);

    if(options.get_bool_option("atomicity-check"))
    {
      unsigned int globals = get_expr_number_globals(guard_expr);
      if(globals > 0)
        break_globals2assignments(guard_expr, tmp_cases,code.location());
    }

    goto_programt::targett x=tmp_cases.add_instruction();
    x->make_goto(it->first);
    migrate_expr(guard_expr, x->guard);
    x->location=case_ops.front().find_location();
  }

  {
    goto_programt::targett d_jump=tmp_cases.add_instruction();
    d_jump->make_goto(targets.default_target);
    d_jump->location=targets.default_target->location;
  }

  dest.destructive_append(sideeffects);
  dest.destructive_append(tmp_cases);
  dest.destructive_append(tmp);
  dest.destructive_append(tmp_z);

  // restore old targets
  targets.restore(old_targets);
}

/*******************************************************************\

Function: goto_convertt::convert_break

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_break(
  const code_breakt &code,
  goto_programt &dest)
{
  if(!targets.break_set)
  {
    err_location(code);
    throw "break without target";
  }

  goto_programt::targett t=dest.add_instruction();
  t->make_goto(targets.break_target);
  t->location=code.location();

  if (current_block != NULL)
    current_block->has_break = true;
}

/*******************************************************************\

Function: goto_convertt::convert_return

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_return(
  const code_returnt &code,
  goto_programt &dest)
{
  if(!targets.return_set)
  {
    err_location(code);
    throw "return without target";
  }

  if(code.operands().size()!=0 &&
     code.operands().size()!=1)
  {
    err_location(code);
    throw "return takes none or one operand";
  }

  code_returnt new_code(code);

  if(new_code.has_return_value())
  {
    goto_programt sideeffects;
    remove_sideeffects(new_code.return_value(), sideeffects);
    dest.destructive_append(sideeffects);

    if(options.get_bool_option("atomicity-check"))
    {
      unsigned int globals = get_expr_number_globals(new_code.return_value());
      if(globals > 0)
        break_globals2assignments(new_code.return_value(), dest,code.location());
    }
  }

  if(targets.return_value)
  {
    if(!new_code.has_return_value())
    {
      err_location(new_code);
      throw "function must return value";
    }
  }
  else
  {
    if(new_code.has_return_value() &&
       new_code.return_value().type().id()!="empty")
    {
      err_location(new_code);
      throw "function must not return value";
    }
  }

  goto_programt::targett t=dest.add_instruction();
  t->make_return();
  migrate_expr(new_code, t->code);
  t->location=new_code.location();
}

/*******************************************************************\

Function: goto_convertt::convert_continue

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_continue(
  const code_continuet &code,
  goto_programt &dest)
{
  if(!targets.continue_set)
  {
    err_location(code);
    throw "continue without target";
  }

  goto_programt::targett t=dest.add_instruction();
  t->make_goto(targets.continue_target);
  t->location=code.location();
}

/*******************************************************************\

Function: goto_convertt::convert_goto

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_goto(
  const codet &code,
  goto_programt &dest)
{
  goto_programt::targett t=dest.add_instruction();
  t->make_goto();
  t->location=code.location();
  migrate_expr(code, t->code);

  // remember it to do target later
  targets.gotos.insert(t);
}

/*******************************************************************\

Function: goto_convertt::convert_non_deterministic_goto

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_non_deterministic_goto(
  const codet &code,
  goto_programt &dest)
{
  convert_goto(code, dest);
}

/*******************************************************************\

Function: goto_convertt::convert_atomic_begin

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_atomic_begin(
  const codet &code,
  goto_programt &dest)
{
  if(code.operands().size()!=0)
  {
    err_location(code);
    throw "atomic_begin expects no operands";
  }


  copy(code, ATOMIC_BEGIN, dest);
}

/*******************************************************************\

Function: goto_convertt::convert_atomic_end

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_atomic_end(
  const codet &code,
  goto_programt &dest)
{
  if(code.operands().size()!=0)
  {
    err_location(code);
    throw "atomic_end expects no operands";
  }

  copy(code, ATOMIC_END, dest);
}

/*******************************************************************\

Function: goto_convertt::generate_ifthenelse

  Inputs:

 Outputs:

 Purpose: if(guard) goto target;

\*******************************************************************/

void goto_convertt::generate_ifthenelse(
  const exprt &guard,
  goto_programt &true_case,
  goto_programt &false_case,
  const locationt &location,
  goto_programt &dest)
{
  if(true_case.instructions.empty() &&
     false_case.instructions.empty())
    return;

  // do guarded gotos directly
  if(false_case.instructions.empty() &&
     true_case.instructions.size()==1 &&
     true_case.instructions.back().is_goto() &&
     is_constant_bool2t(true_case.instructions.back().guard) &&
     to_constant_bool2t(true_case.instructions.back().guard).constant_value)
  {
    migrate_expr(guard, true_case.instructions.back().guard);
    dest.destructive_append(true_case);
    return;
  }

  if(true_case.instructions.empty())
    return generate_ifthenelse(
      gen_not(guard), false_case, true_case, location, dest);

  bool has_else=!false_case.instructions.empty();

  //    if(c) P;
  //--------------------
  // v: if(!c) goto z;
  // w: P;
  // z: ;

  //    if(c) P; else Q;
  //--------------------
  // v: if(!c) goto y;
  // w: P;
  // x: goto z;
  // y: Q;
  // z: ;

  // do the x label
  goto_programt tmp_x;
  goto_programt::targett x=tmp_x.add_instruction();

  // do the z label
  goto_programt tmp_z;
  goto_programt::targett z=tmp_z.add_instruction();
  z->make_skip();

  // y: Q;
  goto_programt tmp_y;
  goto_programt::targett y;
  if(has_else)
  {
    tmp_y.swap(false_case);
    y=tmp_y.instructions.begin();
  }

  // v: if(!c) goto z/y;
  goto_programt tmp_v;
  generate_conditional_branch(
    gen_not(guard), has_else?y:z, location, tmp_v);

  // w: P;
  goto_programt tmp_w;
  tmp_w.swap(true_case);

  // x: goto z;
  x->make_goto(z);

  dest.destructive_append(tmp_v);
  dest.destructive_append(tmp_w);

  if(has_else)
  {
    dest.destructive_append(tmp_x);
    dest.destructive_append(tmp_y);
  }

  dest.destructive_append(tmp_z);
}

/*******************************************************************\

Function: goto_convertt::get_cs_member

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::get_cs_member(
  exprt &expr,
  exprt &result,
  const typet &type,
  bool &found)
{
  DEBUGLOC;

  found=false;
  std::string identifier;

  identifier = "cs$"+i2string(current_block->state_counter);

  exprt lhs_struct("symbol", current_block->state);
  lhs_struct.identifier(identifier);

  exprt new_expr(exprt::member, type);
  new_expr.reserve_operands(1);
  new_expr.copy_to_operands(lhs_struct);
  new_expr.identifier(expr.get_string("identifier"));
  new_expr.component_name(expr.get_string("identifier"));

  assert(!new_expr.get_string("component_name").empty());

  const struct_typet &struct_type = to_struct_type(lhs_struct.type());
  const struct_typet::componentst &components = struct_type.components();
  u_int i = 0;

  for (struct_typet::componentst::const_iterator
       it = components.begin();
       it != components.end();
       it++, i++)
  {
    if (it->get("name").compare(new_expr.get_string("component_name")) == 0)
      found=true;
  }

  if (!found)
  {
    new_expr = expr;
    std::cerr << "warning: the symbol '" << expr.get_string("identifier")
  		  << "' at line " << expr.location().get_line()
  		  << " is not a member of the state struct" << std::endl;
  }


  result = new_expr;
}

/*******************************************************************\

Function: goto_convertt::get_new_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::get_new_expr(exprt &expr, exprt &new_expr, bool &found)
{
  DEBUGLOC;

  if (expr.is_symbol())
  {
    if (expr.type().is_pointer() &&
    		expr.type().subtype().is_symbol())
      found = true;
    else
      get_cs_member(expr, new_expr, expr.type(), found);
  }
  else if (expr.is_constant() || expr.is_typecast())
  {
    new_expr = expr;
    found=true;
  }
  else if (expr.is_index())
  {
    exprt array, index;
    assert(expr.operands().size()==2);

    //do we have an index of index?
    if (expr.op0().operands().size() == 2)
      get_new_expr(expr.op0(), array, found); // this should return another index
    else
      get_cs_member(expr.op0(), array, expr.op0().type(), found); //we have one index only

    get_cs_member(expr.op1(), index, expr.op1().type(), found);

    exprt tmp(exprt::index, expr.op0().type());
    tmp.reserve_operands(2);
    tmp.copy_to_operands(array);
    tmp.copy_to_operands(index);
    new_expr = tmp;
  }
  else if (expr.operands().size() == 1)
  {
    get_new_expr(expr.op0(), new_expr, found);
  }
  else if (expr.operands().size() == 2)
  {
    exprt operand0, operand1;
    get_new_expr(expr.op0(), operand0, found);
    get_new_expr(expr.op1(), operand1, found);

    new_expr = gen_binary(expr.id().as_string(), expr.type(), operand0, operand1);

    if (new_expr.op0().is_index())
      assert(new_expr.op0().type().id() == expr.op0().op0().type().id());
    else if (new_expr.op0().type()!=new_expr.op1().type())
   	  new_expr.op1().make_typecast(new_expr.op0().type());
    else
      assert(new_expr.op0().type()==new_expr.op1().type());
  }
  else
  {
    std::cerr << "warning: the expression '" << expr.pretty()
  		  << "' is not supported yet" << std::endl;
    assert(0);
  }

  if (!found) new_expr = expr;
}

/*******************************************************************\

Function: goto_convertt::replace_ifthenelse

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::replace_ifthenelse(
		exprt &expr)
{
  DEBUGLOC;

  bool found=false;

  if(expr.id()=="constant")
    return;

  if (expr.operands().size()==0 || expr.operands().size() == 1)
  {
    exprt new_expr;
    if (expr.operands().size()==0 || expr.is_typecast())
      get_new_expr(expr, new_expr, found);
    else if (expr.operands().size())
      get_new_expr(expr.op0(), new_expr, found);

    if (!found) std::cout << "failed" << std::endl;
    assert(found);

    expr = new_expr;
  }
  else
  {
    assert(expr.operands().size()==2);

    if(expr.has_operands())
    {
      exprt::operandst::iterator it = expr.operands().begin();
      for( ; it != expr.operands().end(); ++it)
        replace_ifthenelse(*it);
      return;
    }

    nondet_varst::const_iterator result_op0 = nondet_vars.find(expr.op0());
    nondet_varst::const_iterator result_op1 = nondet_vars.find(expr.op1());

    if (result_op0 != nondet_vars.end() && result_op1 != nondet_vars.end())
      return;
    else if (expr.op0().is_constant() || expr.op1().is_constant()) {
      if (result_op0 != nondet_vars.end() || result_op1 != nondet_vars.end())
        return;
    }

    loop_varst::const_iterator cache_result =
      current_block->loop_vars.find(expr.op0());
    if (cache_result == current_block->loop_vars.end())
      return;

    assert(expr.op0().type() == expr.op1().type());

    exprt new_expr1, new_expr2;

    get_new_expr(expr.op0(), new_expr1, found);
    found=false;
    get_new_expr(expr.op1(), new_expr2, found);

    if (expr.op0().is_index())
      assert(new_expr1.type().id() == expr.op0().op0().type().id());
    else if (expr.op1().is_index())
      assert(new_expr2.type().id() == expr.op1().op0().type().id());
    else
      if (new_expr1.type().id() != new_expr2.type().id() ||
          new_expr1.type().width()!=new_expr2.type().width())
        new_expr2.make_typecast(new_expr1.type());

    expr = gen_binary(expr.id().as_string(), bool_typet(), new_expr1, new_expr2);
  }
}

/*******************************************************************\

Function: goto_convertt::convert_ifthenelse

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::convert_ifthenelse(
  const codet &code,
  goto_programt &dest)
{
  set_ifthenelse_block(true);

  if(code.operands().size()!=2 &&
    code.operands().size()!=3)
  {
    err_location(code);
    throw "ifthenelse takes two or three operands";
  }

  bool has_else=
    code.operands().size()==3 &&
    !code.op2().is_nil();

  const locationt &location=code.location();

  // convert 'then'-branch
  goto_programt tmp_op1;
  convert(to_code(code.op1()), tmp_op1);

  goto_programt tmp_op2;

  if(has_else)
    convert(to_code(code.op2()), tmp_op2);

  exprt tmp_guard;
  if (options.get_bool_option("control-flow-test")
    && code.op0().id() != "notequal" && code.op0().id() != "symbol"
    && code.op0().id() != "typecast" && code.op0().id() != "="
    && !options.get_bool_option("deadlock-check"))
  {
    symbolt &new_symbol=new_cftest_symbol(code.op0().type());
    irept irep;
    new_symbol.to_irep(irep);

    codet assignment("assign");
    assignment.reserve_operands(2);
    assignment.copy_to_operands(symbol_expr(new_symbol));
    assignment.copy_to_operands(code.op0());
    assignment.location() = code.op0().find_location();
    copy(assignment, ASSIGN, dest);

    tmp_guard=symbol_expr(new_symbol);
  }
  else if (code.op0().statement() == "decl-block")
  {
    exprt lhs(code.op0().op0().op0());
    lhs.location()=code.op0().op0().location();
    exprt rhs(code.op0().op0().op1());

    rhs.type()=code.op0().op0().op1().type();

    codet assignment("assign");
    assignment.copy_to_operands(lhs);
    assignment.move_to_operands(rhs);
    assignment.location()=lhs.location();
    convert(assignment, dest);

    tmp_guard=assignment.op0();
    if (!tmp_guard.type().is_bool())
      tmp_guard.make_typecast(bool_typet());
  }
  else
    tmp_guard=code.op0();

  remove_sideeffects(tmp_guard, dest);
  if (inductive_step && (current_block != NULL) && current_block->active)
  {
    replace_ifthenelse(tmp_guard);

    if (!tmp_guard.type().is_bool())
      tmp_guard.make_typecast(bool_typet());
  }

  generate_ifthenelse(tmp_guard, tmp_op1, tmp_op2, location, dest);
  set_ifthenelse_block(false);
}

/*******************************************************************\

Function: goto_convertt::collect_operands

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::collect_operands(
  const exprt &expr,
  const irep_idt &id,
  std::list<exprt> &dest)
{
  if(expr.id()!=id)
  {
    dest.push_back(expr);
  }
  else
  {
    // left-to-right is important
    forall_operands(it, expr)
      collect_operands(*it, id, dest);
  }
}

/*******************************************************************\

Function: goto_convertt::generate_conditional_branch

  Inputs:

 Outputs:

 Purpose: if(guard) goto target;

\*******************************************************************/

void goto_convertt::generate_conditional_branch(
  const exprt &guard,
  goto_programt::targett target_true,
  const locationt &location,
  goto_programt &dest)
{

  if(!has_sideeffect(guard))
  {
    exprt g = guard;
    if(options.get_bool_option("atomicity-check"))
    {
      unsigned int globals = get_expr_number_globals(g);
      if(globals > 0)
        break_globals2assignments(g, dest,location);
    }
    // this is trivial
    goto_programt::targett t=dest.add_instruction();
    t->make_goto(target_true);
    migrate_expr(g, t->guard);
    t->location=location;
    return;
  }

  // if(guard) goto target;
  //   becomes
  // if(guard) goto target; else goto next;
  // next: skip;

  goto_programt tmp;
  goto_programt::targett target_false=tmp.add_instruction();
  target_false->make_skip();

  generate_conditional_branch(guard, target_true, target_false, location, dest);

  dest.destructive_append(tmp);
}

/*******************************************************************\

Function: goto_convertt::generate_conditional_branch

  Inputs:

 Outputs:

 Purpose: if(guard) goto target;

\*******************************************************************/

void goto_convertt::generate_conditional_branch(
  const exprt &guard,
  goto_programt::targett target_true,
  goto_programt::targett target_false,
  const locationt &location,
  goto_programt &dest)
{
  if(guard.id()=="not")
  {
    assert(guard.operands().size()==1);
    // swap targets
    generate_conditional_branch(
      guard.op0(), target_false, target_true, location, dest);
    return;
  }

  if(!has_sideeffect(guard))
  {
	exprt g = guard;
	if(options.get_bool_option("atomicity-check"))
	{
	  unsigned int globals = get_expr_number_globals(g);
	  if(globals > 0)
		break_globals2assignments(g, dest,location);
	}

    // this is trivial
    goto_programt::targett t_true=dest.add_instruction();
    t_true->make_goto(target_true);
    migrate_expr(guard, t_true->guard);
    t_true->location=location;

    goto_programt::targett t_false=dest.add_instruction();
    t_false->make_goto(target_false);
    t_false->guard = true_expr;
    t_false->location=location;
    return;
  }

  if(guard.is_and())
  {
    // turn
    //   if(a && b) goto target_true; else goto target_false;
    // into
    //    if(!a) goto target_false;
    //    if(!b) goto target_false;
    //    goto target_true;

    std::list<exprt> op;
    collect_operands(guard, guard.id(), op);

    forall_expr_list(it, op)
      generate_conditional_branch(
        gen_not(*it), target_false, location, dest);

    goto_programt::targett t_true=dest.add_instruction();
    t_true->make_goto(target_true);
    t_true->guard = true_expr;
    t_true->location=location;

    return;
  }
  else if(guard.id()=="or")
  {
    // turn
    //   if(a || b) goto target_true; else goto target_false;
    // into
    //   if(a) goto target_true;
    //   if(b) goto target_true;
    //   goto target_false;

    std::list<exprt> op;
    collect_operands(guard, guard.id(), op);

    forall_expr_list(it, op)
      generate_conditional_branch(
        *it, target_true, location, dest);

    goto_programt::targett t_false=dest.add_instruction();
    t_false->make_goto(target_false);
    t_false->guard = true_expr;
    t_false->location=guard.location();

    return;
  }

  exprt cond=guard;
  remove_sideeffects(cond, dest);

  if(options.get_bool_option("atomicity-check"))
  {
    unsigned int globals = get_expr_number_globals(cond);
	if(globals > 0)
	  break_globals2assignments(cond, dest,location);
  }

  goto_programt::targett t_true=dest.add_instruction();
  t_true->make_goto(target_true);
  migrate_expr(cond, t_true->guard);
  t_true->location=guard.location();

  goto_programt::targett t_false=dest.add_instruction();
  t_false->make_goto(target_false);
  t_false->guard = true_expr;
  t_false->location=guard.location();
}

/*******************************************************************\

Function: goto_convertt::get_string_constant

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

const std::string &goto_convertt::get_string_constant(
  const exprt &expr)
{
  if(expr.id()=="typecast" &&
     expr.operands().size()==1)
    return get_string_constant(expr.op0());

  if(!expr.is_address_of() ||
     expr.operands().size()!=1 ||
     expr.op0().id()!="index" ||
     expr.op0().operands().size()!=2 ||
     expr.op0().op0().id()!="string-constant")
  {
    err_location(expr);
    str << "expected string constant, but got: "
          << expr.pretty() << std::endl;
    throw 0;
  }

  return expr.op0().op0().value().as_string();
}

/*******************************************************************\

Function: goto_convertt::new_tmp_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

symbolt &goto_convertt::new_tmp_symbol(const typet &type)
{
  symbolt new_symbol;
  symbolt *symbol_ptr;

  do {
    new_symbol.base_name="tmp$"+i2string(++temporary_counter);
    new_symbol.name=tmp_symbol_prefix+id2string(new_symbol.base_name);
    new_symbol.lvalue=true;
    new_symbol.type=type;
  } while (context.move(new_symbol, symbol_ptr));

  tmp_symbols.push_back(symbol_ptr->name);

  return *symbol_ptr;
}

/*******************************************************************\

Function: goto_convertt::new_cftest_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

symbolt &goto_convertt::new_cftest_symbol(const typet &type)
{
  static int cftest_counter=0;
  symbolt new_symbol;
  symbolt *symbol_ptr;

  do {
    new_symbol.base_name="cftest$"+i2string(++cftest_counter);
    new_symbol.name=tmp_symbol_prefix+id2string(new_symbol.base_name);
    new_symbol.lvalue=true;
    new_symbol.type=type;
  } while (context.move(new_symbol, symbol_ptr));

  tmp_symbols.push_back(symbol_ptr->name);

  return *symbol_ptr;
}

/*******************************************************************\

Function: goto_convertt::guard_program

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_convertt::guard_program(
  const guardt &guard,
  goto_programt &dest)
{
  if(guard.is_true()) return;

  // the target for the GOTO
  goto_programt::targett t=dest.add_instruction(SKIP);

  goto_programt tmp;
  tmp.add_instruction(GOTO);
  tmp.instructions.front().targets.push_back(t);
  exprt guardexpr = migrate_expr_back(guard.as_expr());
  guardexpr.make_not();
  migrate_expr(guardexpr, tmp.instructions.front().guard);
  tmp.destructive_append(dest);

  tmp.swap(dest);
}

void goto_convertt::push_new_loop_block()
{
  // Add new block to stack
  loop_block *block = new loop_block(total_states, global_vars);
  loop_stack.push(block);

  // Update current_block reference
  current_block = block;

  // Increment total number of states
  ++total_states;
}

void goto_convertt::pop_loop_block()
{
  assert(loop_stack.size() > 0);

  // Create symbol for the state$vector
  symbolt *symbol_ptr=NULL;
  unsigned int i = current_block->state_counter;

  symbolt state_symbol;
  state_symbol.name="c::state$vector"+i2string(i);
  state_symbol.base_name="state$vector"+i2string(i);
  state_symbol.is_type=true;
  state_symbol.type=current_block->state;
  state_symbol.mode="C";
  state_symbol.module="main";
  state_symbol.pretty_name="struct state$vector"+i2string(i);

  context.move(state_symbol, symbol_ptr);

  // Create new symbol for this state
  // First is kindice
  symbolt kindice_symbol;
  kindice_symbol.name="kindice$"+i2string(i);
  kindice_symbol.base_name="kindice$"+i2string(i);
  kindice_symbol.type=uint_type();
  kindice_symbol.static_lifetime=true;
  kindice_symbol.lvalue=true;

  context.move(kindice_symbol, symbol_ptr);

  // Then state_vector s
  // Its type is incomplete array
  typet incomplete_array_type("incomplete_array");
  incomplete_array_type.subtype() = struct_typet();

  symbolt state_vector_symbol;
  state_vector_symbol.name="s$"+i2string(i);
  state_vector_symbol.base_name="s$"+i2string(i);
  state_vector_symbol.type=incomplete_array_type;
  state_vector_symbol.static_lifetime=true;
  state_vector_symbol.lvalue=true;

  context.move(state_vector_symbol, symbol_ptr);

  // Finally, the current state cs
  typet state_type("struct");
  state_type.tag("state$vector"+i2string(i));

  symbolt current_state_symbol;
  current_state_symbol.name="cs$"+i2string(i);
  current_state_symbol.base_name="cs$"+i2string(i);
  current_state_symbol.type=state_type;
  current_state_symbol.static_lifetime=true;
  current_state_symbol.lvalue=true;

  context.move(current_state_symbol, symbol_ptr);

  // Update flag to disable inductive step
  if(current_block->active)
    disable_inductive_step = false;

  delete current_block;

  // Pop block from stack
  loop_stack.pop();

  // Update reference
  if(loop_stack.size())
    current_block = loop_stack.top();
  else
    current_block = NULL;
}

bool goto_convertt::is_inductive_step_active()
{
  return inductive_step && (current_block != NULL) && current_block->active;
}
