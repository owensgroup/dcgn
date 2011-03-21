#include <dcgn/OutputStream.h>
#include <cstring>
#include <cstdio>

namespace dcgn
{
  void OutputStream::write(const char * const s, const int start, const int end)
  {
    if (end - start + plen > maxLen)
    {
      flush();
    }
    memcpy(printed + plen, s + start, end - start);
    plen += end - start;
  }
  void OutputStream::init(const int maxSize)
  {
    maxLen = 4096;
    while (maxLen < maxSize) maxLen <<= 1;
    printed = new char[maxLen];
    ptr = new char[maxLen];
    memory = reinterpret_cast<void * >(ptr);
    fmt = 0;
    plen = 0;
  }
  void OutputStream::destroy()
  {
    delete [] reinterpret_cast<char * >(memory);
    delete [] printed;
  }

  OutputStream & OutputStream::format(const char * const s)
  {
    ptr = reinterpret_cast<char * >(memory);
    fmt = s;
    return * this;
  }
  OutputStream & OutputStream::arg(const          char          p)
  {
    return arg(static_cast<int>(p));
  }
  OutputStream & OutputStream::arg(const unsigned char          p)
  {
    return arg(static_cast<unsigned int>(p));
  }
  OutputStream & OutputStream::arg(const          short         p)
  {
    return arg(static_cast<int>(p));
  }
  OutputStream & OutputStream::arg(const unsigned short         p)
  {
    return arg(static_cast<unsigned int>(p));
  }
  OutputStream & OutputStream::arg(const          int           p)
  {
    *reinterpret_cast<int * >(ptr) = p;
    ptr += sizeof(int);
    return * this;
  }
  OutputStream & OutputStream::arg(const unsigned int           p)
  {
    *reinterpret_cast<unsigned int * >(ptr) = p;
    ptr += sizeof(unsigned int);
    return * this;
  }
  OutputStream & OutputStream::arg(const          long long     p)
  {
    *reinterpret_cast<long long * >(ptr) = p;
    ptr += sizeof(long long);
    return * this;
  }
  OutputStream & OutputStream::arg(const unsigned long long     p)
  {
    *reinterpret_cast<unsigned long long * >(ptr) = p;
    ptr += sizeof(unsigned long long);
    return * this;
  }
  OutputStream & OutputStream::arg(const float                  p)
  {
    return arg(static_cast<double>(p));
  }
  OutputStream & OutputStream::arg(const double                 p)
  {
    *reinterpret_cast<double * >(ptr) = p;
    ptr += sizeof(double);
    return * this;
  }
  OutputStream & OutputStream::arg(const          char * const  p)
  {
    *reinterpret_cast<const char ** >(ptr) = p;
    ptr += sizeof(const char * );
    return * this;
  }
  OutputStream & OutputStream::arg(const unsigned char * const  p)
  {
    *reinterpret_cast<unsigned const char ** >(ptr) = p;
    ptr += sizeof(unsigned const char * );
    return * this;
  }
  OutputStream & OutputStream::args(const void * const arguments, const int len)
  {
    memcpy(ptr, arguments, len);
    ptr += len;
    return * this;
  }
  OutputStream & OutputStream::finish()
  {
    if (!fmt) return * this;
    char formatString[1024], formatted[1024];
    int lastByte = 0, i;
    bool done;
    ptr = reinterpret_cast<char * >(memory);
    for (i = 0; fmt[i]; )
    {
      if (fmt[i] == '%')
      {
        int width = -1, prec = -1;
        int percLoc = i;

        bool foundDot = false, foundWidth = false, foundPrec = false;
        write(fmt, lastByte, i);
        done = false;
        ++i;
        while (!done)
        {
          switch (fmt[i])
          {
          case '.':
            foundDot = true;
            ++i;
            break;
          case '*':
            if (!foundDot)
            {
              width = *reinterpret_cast<int * >(ptr);
              ptr += sizeof(int);
              foundWidth = true;
            }
            else
            {
              prec = *reinterpret_cast<int * >(ptr);
              ptr += sizeof(int);
              foundPrec = true;
            }
            ++i;
            break;
          case 0:
            write(fmt, percLoc, i);
            done = true;
            break;
          case 'c':
            {
              int ival = *reinterpret_cast<int * >(ptr);
              ptr += sizeof(int);
              memcpy(formatString, fmt + percLoc, i + 1 - percLoc);
              formatString[i + 1 - percLoc] = 0;
              if      (foundWidth && foundPrec) sprintf(formatted, formatString, width, prec, ival);
              else if (foundWidth)              sprintf(formatted, formatString, width,       ival);
              else if (foundPrec)               sprintf(formatted, formatString,        prec, ival);
              else                              sprintf(formatted, formatString,              ival);
              write(formatted, 0, strlen(formatted));
              ++i;
              done = true;
            }
            break;
          case 'd':
          case 'i':
          case 'o':
          case 'u':
          case 'x':
          case 'X':
            if (i - percLoc > 3 && fmt[i - 1] == 'l' && fmt[i - 2] == 'l')
            {
              long long lval = *reinterpret_cast<long long * >(ptr);
              ptr += sizeof(long long);
              memcpy(formatString, fmt + percLoc, i + 1 - percLoc);
              formatString[i + 1 - percLoc] = 0;
              if      (foundWidth && foundPrec) sprintf(formatted, formatString, width, prec, lval);
              else if (foundWidth)              sprintf(formatted, formatString, width,       lval);
              else if (foundPrec)               sprintf(formatted, formatString,        prec, lval);
              else                              sprintf(formatted, formatString,              lval);
              write(formatted, 0, strlen(formatted));
              ++i;
              done = true;
            }
            else
            {
              int ival = *reinterpret_cast<int * >(ptr);
              ptr += sizeof(int);
              memcpy(formatString, fmt + percLoc, i + 1 - percLoc);
              formatString[i + 1 - percLoc] = 0;
              if      (foundWidth && foundPrec) sprintf(formatted, formatString, width, prec, ival);
              else if (foundWidth)              sprintf(formatted, formatString, width,       ival);
              else if (foundPrec)               sprintf(formatted, formatString,        prec, ival);
              else                              sprintf(formatted, formatString,              ival);
              write(formatted, 0, strlen(formatted));
              ++i;
              done = true;
            }
            break;
          case 'n':
            ptr += sizeof(int * );
            ++i;
            done = true;
            break;
          case 'e':
          case 'E':
          case 'f':
          case 'g':
          case 'G':
            if (i - percLoc > 2 && fmt[i - 1] == 'L')
            {
              long double dval = *reinterpret_cast<long double * >(ptr);
              ptr += sizeof(long double);
              memcpy(formatString, fmt + percLoc, i + 1 - percLoc);
              formatString[i + 1 - percLoc] = 0;
              if      (foundWidth && foundPrec) sprintf(formatted, formatString, width, prec, dval);
              else if (foundWidth)              sprintf(formatted, formatString, width,       dval);
              else if (foundPrec)               sprintf(formatted, formatString,        prec, dval);
              else                              sprintf(formatted, formatString,              dval);
              write(formatted, 0, strlen(formatted));
              ++i;
              done = true;
            }
            else
            {
              double dval = *reinterpret_cast<int * >(ptr);
              ptr += sizeof(double);
              memcpy(formatString, fmt + percLoc, i + 1 - percLoc);
              formatString[i + 1 - percLoc] = 0;
              if      (foundWidth && foundPrec) sprintf(formatted, formatString, width, prec, dval);
              else if (foundWidth)              sprintf(formatted, formatString, width,       dval);
              else if (foundPrec)               sprintf(formatted, formatString,        prec, dval);
              else                              sprintf(formatted, formatString,              dval);
              write(formatted, 0, strlen(formatted));
              ++i;
              done = true;
            }
            break;
          case 's':
          case 'p':
            {
              const char * sval = *reinterpret_cast<const char ** >(ptr);
              ptr += sizeof(char * );
              memcpy(formatString, fmt + percLoc, i + 1 - percLoc);
              formatString[i + 1 - percLoc] = 0;
              if      (foundWidth && foundPrec) sprintf(formatted, formatString, width, prec, sval);
              else if (foundWidth)              sprintf(formatted, formatString, width,       sval);
              else if (foundPrec)               sprintf(formatted, formatString,        prec, sval);
              else                              sprintf(formatted, formatString,              sval);
              write(formatted, 0, strlen(formatted));
              ++i;
              done = true;
            }
            break;
          default:
            ++i;
            break;
          }
        }
        lastByte = i;
      }
      else
      {
        ++i;
      }
    }
    write(fmt, lastByte, i);
    ptr = reinterpret_cast<char * >(memory);
    fmt = 0;
    return * this;
  }
  OutputStream & OutputStream::flush()
  {
    finish();
    if (plen > 0)
    {
      fwrite(printed, plen, 1, stderr);
      fflush(stderr);
      plen = 0;
    }
    return * this;
  }
}
