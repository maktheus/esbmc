/*
 * llvmtypecheck.cpp
 *
 *  Created on: Jul 23, 2015
 *      Author: mramalho
 */

#include "llvm_convert.h"

#include <arith_tools.h>
#include <bitvector.h>
#include <expr_util.h>

#include <ansi-c/c_types.h>
#include <ansi-c/ansi_c_expr.h>

#include <sstream>

void llvm_convertert::convert_character_literal(
  const clang::CharacterLiteral char_literal,
  exprt &dest)
{
  typet type;
  get_type(char_literal.getType(), type);

  dest =
    constant_exprt(
      integer2binary(char_literal.getValue(), bv_width(type)),
      integer2string(char_literal.getValue()),
      type);
}

void llvm_convertert::convert_string_literal(
  const clang::StringLiteral string_literal,
  exprt &dest)
{
  typet type;
  get_type(string_literal.getType(), type);

  string_constantt string;
  string.set_value(string_literal.getString().str());

  index_exprt zero_index(string, gen_zero(int_type()), type);
  dest = address_of_exprt(zero_index);
}

void llvm_convertert::convert_integer_literal(
  llvm::APInt val,
  typet type,
  exprt &dest)
{
  assert(type.is_unsignedbv() || type.is_signedbv());

  exprt the_val;
  if (type.is_unsignedbv())
  {
    the_val =
      constant_exprt(
        integer2binary(val.getZExtValue(), bv_width(type)),
        integer2string(val.getZExtValue()),
        type);
  }
  else
  {
    the_val =
      constant_exprt(
        integer2binary(val.getSExtValue(), bv_width(type)),
        integer2string(val.getSExtValue()),
        type);
  }

  dest.swap(the_val);
}

void llvm_convertert::convert_float_literal(
  llvm::APFloat val,
  typet type,
  exprt &dest)
{
  bool ignored;
  val.convert(llvm::APFloat::IEEEdouble, llvm::APFloat::rmTowardZero, &ignored);

  std::ostringstream strs;
  strs << val.convertToDouble();

  mp_integer significand;
  mp_integer exponent;
  bool is_float, is_long;

  parse_float(strs.str(), significand, exponent, is_float, is_long);

  dest = constant_exprt(type);

  if(config.ansi_c.use_fixed_for_float)
  {
    unsigned width=atoi(dest.type().width().c_str());
    unsigned fraction_bits;
    const std::string &integer_bits=dest.type().integer_bits().as_string();

    if(integer_bits=="")
      fraction_bits=width/2;
    else
      fraction_bits=width-atoi(integer_bits.c_str());

    mp_integer factor=mp_integer(1)<<fraction_bits;
    mp_integer value=significand*factor;

    if(exponent<0)
      value/=power(10, -exponent);
    else
      value*=power(10, exponent);

    dest.value(integer2binary(value, width));
  }
  else
  {
    std::cerr << "floatbv unsupported, sorry" << std::endl;
    abort();
  }

}

void llvm_convertert::parse_float(
  const std::string &src,
  mp_integer &significand,
  mp_integer &exponent,
  bool &is_float, bool &is_long)
{
  // <INITIAL>{digits}{dot}{digits}{exponent}?{floatsuffix}?
  // <INITIAL>{digits}{dot}{exponent}?{floatsuffix}?
  // <INITIAL>{dot}{digits}{exponent}?{floatsuffix}?
  // <INITIAL>{digits}{exponent}{floatsuffix}?

  const char *p=src.c_str();

  std::string str_whole_number,
              str_fraction_part,
              str_exponent;

  // get whole number part
  while(*p!='.' && *p!=0 && *p!='e' && *p!='E' &&
        *p!='f' && *p!='F' && *p!='l' && *p!='L')
  {
    str_whole_number+=*p;
    p++;
  }

  // skip dot
  if(*p=='.')
    p++;

  // get fraction part
  while(*p!=0 && *p!='e' && *p!='E' &&
         *p!='f' && *p!='F' && *p!='l' && *p!='L')
  {
    str_fraction_part+=*p;
    p++;
  }

  // skip E
  if(*p=='e' || *p=='E')
    p++;

  // skip +
  if(*p=='+')
    p++;

  // get exponent
  while(*p!=0 && *p!='f' && *p!='F' && *p!='l' && *p!='L')
  {
    str_exponent+=*p;
    p++;
  }

  // get flags
  is_float=is_long=false;

  while(*p!=0)
  {
    if(*p=='f' || *p=='F')
      is_float=true;
    else if(*p=='l' || *p=='L')
      is_long=true;

    p++;
  }

  std::string str_number=str_whole_number+
                         str_fraction_part;

  if(str_number.empty())
    significand=0;
  else
    significand=string2integer(str_number);

  if(str_exponent.empty())
    exponent=0;
  else
    exponent=string2integer(str_exponent);

  // adjust exponent
  exponent-=str_fraction_part.size();
}
