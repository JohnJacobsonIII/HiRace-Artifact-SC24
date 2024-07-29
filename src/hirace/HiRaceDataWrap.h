#ifndef HIRACEDATAWRAP_H
#define HIRACEDATAWRAP_H

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __device__
// # define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

enum class Scope {
  Global = 0,
  Block = 1,
  Warp = 2,
  Tid = 3,
};

template <typename T>
class HiRaceDataWrap;

template <typename T>
class HiRaceNumWrap {
public:
  CUDA_CALLABLE_MEMBER HiRaceNumWrap(HiRaceDataWrap<T> &parent) : _parent(parent) {}
  
  CUDA_CALLABLE_MEMBER HiRaceNumWrap(const HiRaceNumWrap<T> &other) = default;
  
  CUDA_CALLABLE_MEMBER operator T() const {
    _parent.log(R);
    return _parent._data[_parent._offset_x];
  }

  CUDA_CALLABLE_MEMBER HiRaceNumWrap& operator++() {
    HiRaceNumWrap temp = *this;
    ++*this; // call prefix operator which logs
    return temp;
  }
  CUDA_CALLABLE_MEMBER HiRaceNumWrap& operator++(int) {
    _parent.log(W);
    _parent._data[_parent._offset_x]++;
    return *this;
  }
  
  CUDA_CALLABLE_MEMBER HiRaceNumWrap& operator--() {
    HiRaceNumWrap temp = *this;
    --*this;
    return temp;
  }
  CUDA_CALLABLE_MEMBER HiRaceNumWrap& operator--(int) {
    _parent.log(W);
    _parent._data[_parent._offset_x]--;
    return *this;
  }

  CUDA_CALLABLE_MEMBER HiRaceNumWrap& operator=(T val) {
    _parent.log(W);
    _parent._data[_parent._offset_x] = val;
    return *this;
  }

  CUDA_CALLABLE_MEMBER HiRaceNumWrap& operator=(HiRaceNumWrap val) {
    return operator=((T)val);
  }

  CUDA_CALLABLE_MEMBER HiRaceNumWrap& operator+=(T val) {
    _parent.log(W);
    _parent._data[_parent._offset_x] += val;
    return *this;
  }

  CUDA_CALLABLE_MEMBER HiRaceNumWrap& operator-=(T val) {
    _parent.log(W);
    _parent._data[_parent._offset_x] -= val;
    return *this;
  }

  CUDA_CALLABLE_MEMBER HiRaceNumWrap& operator*=(T val) {
    _parent.log(W);
    _parent._data[_parent._offset_x] *= val;
    return *this;
  }

  CUDA_CALLABLE_MEMBER HiRaceDataWrap<T>& operator&() {
    return _parent;
  }

  CUDA_CALLABLE_MEMBER HiRaceDataWrap<T> operator*() {
    return *_parent;
  }

  CUDA_CALLABLE_MEMBER T atomicCAS(T compare, T val) {
    _parent.log(GA);
    return ::atomicCAS(_parent._data + _parent._offset_x, compare, val);
    // return *this;
  }

#define WRAP_ATOMIC(name) \
  CUDA_CALLABLE_MEMBER T name(T val) { \
    _parent.log(GA); \
    return ::name(_parent._data + _parent._offset_x, val); \
  }

  WRAP_ATOMIC(atomicAdd)
  WRAP_ATOMIC(atomicSub)
  WRAP_ATOMIC(atomicExch)
  WRAP_ATOMIC(atomicMin)
  WRAP_ATOMIC(atomicMax)
  WRAP_ATOMIC(atomicInc)
  WRAP_ATOMIC(atomicDec)
  WRAP_ATOMIC(atomicAnd)
  WRAP_ATOMIC(atomicOr)
  WRAP_ATOMIC(atomicXor)

#undef WRAP_ATOMIC


private:
  HiRaceDataWrap<T> &_parent;
};

template <typename T>
class HiRaceDataWrap {
public:
  CUDA_CALLABLE_MEMBER HiRaceDataWrap() : _offset_x(0),
                                          _scope(Scope::Tid),
                                          _numWrap(*this) {}

  CUDA_CALLABLE_MEMBER HiRaceDataWrap(T* data) : _data(data),
                                                 _offset_x(0),
                                                 _scope(Scope::Tid),
                                                 _numWrap(*this) {}
  
  CUDA_CALLABLE_MEMBER HiRaceDataWrap(T* data,
                                      unsigned *bcount,
                                      unsigned *wcount,
                                      unsigned *swidx) : 
                                        _data(data),
                                        _offset_x(0),
                                        _bcount(bcount),
                                        _wcount(wcount),
                                        _swidx(swidx),
                                        _scope(Scope::Tid),
                                        _numWrap(*this) {}
  // higher dim call
  CUDA_CALLABLE_MEMBER HiRaceDataWrap(T* data,
                                      hr_shadowt *metadata,
                                      Scope scope,
                                      unsigned *bcount,
                                      unsigned *wcount,
                                      unsigned *swidx,
                                      int dim,
                                      int offset_y,
                                      int offset_z) : 
                                        _data(data),
                                        _metadata(metadata),
                                        _dim(dim),
                                        _offset_x(0),
                                        _offset_y(offset_y),
                                        _offset_z(offset_z),
                                        _bcount(bcount),
                                        _wcount(wcount),
                                        _swidx(swidx),
                                        _scope(scope),
                                        _numWrap(*this) {}
  
  CUDA_CALLABLE_MEMBER void setMembers(T *data,
                                       hr_shadowt *metadata,
                                       Scope scope,
                                       unsigned *bcount,
                                       unsigned *wcount,
                                       unsigned *swidx,
                                       int dim,
                                       int offset_y,
                                       int offset_z) {
    _data = data;
    _metadata = metadata;
    _bcount = bcount;
    _wcount = wcount;
    _swidx = swidx;
    _scope = scope;
    _dim = dim;
    _offset_x = 0;
    _offset_y = offset_y;
    _offset_z = offset_z;
  }
  
  // comparisons
  CUDA_CALLABLE_MEMBER bool operator==(HiRaceDataWrap<T> &other) const { return _data == other._data; }
  CUDA_CALLABLE_MEMBER bool operator!=(HiRaceDataWrap<T> &other) const { return _data != other._data; }
  CUDA_CALLABLE_MEMBER bool operator< (HiRaceDataWrap<T> &other) const { return _data <  other._data; }
  CUDA_CALLABLE_MEMBER bool operator<=(HiRaceDataWrap<T> &other) const { return _data <= other._data; }
  CUDA_CALLABLE_MEMBER bool operator> (HiRaceDataWrap<T> &other) const { return _data >  other._data; }
  CUDA_CALLABLE_MEMBER bool operator>=(HiRaceDataWrap<T> &other) const { return _data >= other._data; }
  
  CUDA_CALLABLE_MEMBER operator HiRaceNumWrap<T>() const { return _numWrap; }
 
  CUDA_CALLABLE_MEMBER HiRaceDataWrap& operator+(int val) {
    _offset_x += val;
    return *this;
  }
  
  CUDA_CALLABLE_MEMBER HiRaceDataWrap& operator-(int val) {
    _offset_x -= val;
    return *this;
  }

  CUDA_CALLABLE_MEMBER HiRaceNumWrap<T> operator*() {
    return _numWrap;
  }
  
  CUDA_CALLABLE_MEMBER HiRaceNumWrap<T>& operator[](int idx) {
    _offset_x = idx;
    return _numWrap;
  }
 
  
  CUDA_CALLABLE_MEMBER void setScope(Scope scope)            { _scope = scope; }
  CUDA_CALLABLE_MEMBER void setData(T* data)                 { _data = data; }
  CUDA_CALLABLE_MEMBER void setMetadata(hr_shadowt *metadata) { _metadata = metadata; }
  CUDA_CALLABLE_MEMBER HiRaceDataWrap& registerCallsite(int lineNo, const char *fileName) {
      _lineNo = lineNo;
      _fileName = fileName;
      return *this;
  }
  
  CUDA_CALLABLE_MEMBER void log(unsigned access_type) {
    int tid = get_globalTID();
    int bid = get_globalBID();
    
    if (_scope == Scope::Tid) { return; }      
    if (bid > (2 << BLOCK_BITS)-2)      _scope = Scope::Tid; // stop logging before overflow
    if (*_bcount > (2 << BCOUNT_BITS)-2) _scope = Scope::Tid; 
    if (*_wcount > (2 << WCOUNT_BITS)-2) _scope = Scope::Tid; 
    
    if (_scope == Scope::Global  
        || (bid == 0 // Only log within monitored scope (i.e. only tracking block races in 1 block.)
            && (_scope == Scope::Block
                || (_scope == Scope::Warp
                    && tid / WARP_SIZE == 0))))
    {
      log_access(
        get_blockTID(),
        bid,
        access_type,
        _bcount,
        _wcount,
        _swidx,
        (_metadata + _offset_x),
        _offset_x,
        _lineNo,
        _fileName);
    }
  }

  CUDA_CALLABLE_MEMBER T atomicCAS(T compare, T val) /*const*/ {
    return _numWrap.atomicCAS(compare, val);
  }

#define WRAP_ATOMIC(name) \
  CUDA_CALLABLE_MEMBER T name(T val) /*const*/ { \
    return _numWrap.name(val); \
  }

  WRAP_ATOMIC(atomicAdd)
  WRAP_ATOMIC(atomicSub)
  WRAP_ATOMIC(atomicExch)
  WRAP_ATOMIC(atomicMin)
  WRAP_ATOMIC(atomicMax)
  WRAP_ATOMIC(atomicInc)
  WRAP_ATOMIC(atomicDec)
  WRAP_ATOMIC(atomicAnd)
  WRAP_ATOMIC(atomicOr)
  WRAP_ATOMIC(atomicXor)

#undef WRAP_ATOMIC

private:
  friend HiRaceNumWrap<T>;
  
  T *_data;
  hr_shadowt *_metadata;
  int _len_x;
  int _len_y;
  int _len_z;
  int _offset_x;
  int _offset_y;
  int _offset_z;
  int _dim;
  unsigned *_bcount;
  unsigned *_wcount;
  unsigned *_swidx;
  Scope _scope;
  int _lineNo;
  const char *_fileName;
  HiRaceNumWrap<T> _numWrap;
 
  CUDA_CALLABLE_MEMBER int get_threadsPerBlock() const {
    return blockDim.x * blockDim.y * blockDim.z;
  }
  
  // linearized thread ID's relative to a single block
  CUDA_CALLABLE_MEMBER int get_blockTID() const { 
    return threadIdx.x
         + threadIdx.y * blockDim.x
         + threadIdx.z * (blockDim.x * blockDim.y);
  }
   
  // linearized block ID's
  CUDA_CALLABLE_MEMBER int get_globalBID() const { 
    return blockIdx.x
         + blockIdx.y * gridDim.x
         + blockIdx.z * (gridDim.x * gridDim.y);
  }
  
  // linearized thread ID's
  CUDA_CALLABLE_MEMBER int get_globalTID() const {
    int tpb = get_threadsPerBlock();
    
    return get_blockTID()
         + (blockIdx.x * tpb) // add a full block for each x step in the grid
         + (blockIdx.y * gridDim.x * tpb) // a full row for each y
         + (blockIdx.z * gridDim.x * gridDim.y * tpb); // and a square for z
  }
  
  
  
}; // HiRaceDataWrap<T>

template <typename T, int xDim, int yDim>
class HiRaceDataWrap2D {
public:
  CUDA_CALLABLE_MEMBER HiRaceDataWrap2D(T (*data)[yDim]) : _data(data),
                                                           _scope(Scope::Tid),
                                                           _dim(2),
                                                           _offset_y(0),
                                                           _offset_z(0),
                                                           _xWrapper() {}
  
  CUDA_CALLABLE_MEMBER HiRaceDataWrap2D(T (*data)[yDim],
                                        hr_shadowt (*metadata)[yDim],
                                        Scope scope,
                                        unsigned *bcount,
                                        unsigned *wcount,
                                        unsigned *swidx,
                                        int dim,
                                        int offset_z) : 
                                          _data(data),
                                          _metadata(metadata),
                                          _bcount(bcount),
                                          _wcount(wcount),
                                          _swidx(swidx),
                                          _scope(scope),
                                          _dim(dim),
                                          _offset_y(0),
                                          _offset_z(offset_z),
                                          _xWrapper() {}

  CUDA_CALLABLE_MEMBER void setMembers(T (*data)[yDim],
                                        hr_shadowt (*metadata)[yDim],
                                        Scope scope,
                                        unsigned *bcount,
                                        unsigned *wcount,
                                        unsigned *swidx,
                                        int dim,
                                        int offset_z) {
    _data = data;
    _metadata = metadata;
    _bcount = bcount;
    _wcount = wcount;
    _swidx = swidx;
    _scope = scope;
    _dim = dim;
    _offset_y = 0;
    _offset_z = offset_z;
  }
  
  CUDA_CALLABLE_MEMBER operator T*() { return _data; }
  
  CUDA_CALLABLE_MEMBER HiRaceDataWrap<T>& operator[](int idx) {
    _offset_y = idx;
    if (_scope == Scope::Tid) {
      _xWrapper.setMembers(_data[idx],
                           nullptr,
                           _scope,
                           _bcount,
                           _wcount,
                           _swidx,
                           _dim,
                           _offset_y,
                           _offset_z);
    }
    else {
      _xWrapper.setMembers(_data[idx],
                           _metadata[idx],
                           _scope,
                           _bcount,
                           _wcount,
                           _swidx,
                           _dim,
                           _offset_y,
                           _offset_z);
    }
    
    return _xWrapper.registerCallsite(_lineNo, _fileName);
  }
  
  CUDA_CALLABLE_MEMBER HiRaceDataWrap2D& registerCallsite(int lineNo, const char *fileName) {
      _lineNo = lineNo;
      _fileName = fileName;
      return *this;
  }

private:
  T (*_data)[yDim];
  hr_shadowt (*_metadata)[yDim];
  
  HiRaceDataWrap<T> _xWrapper;
  
  unsigned *_bcount;
  unsigned *_wcount;
  unsigned *_swidx;
  Scope _scope;
  int _dim;
  int _offset_y;
  int _offset_z;
  int _lineNo;
  const char *_fileName;
}; // HiRaceDataWrap2D<T>

template <typename T, int xDim, int yDim, int zDim>
class HiRaceDataWrap3D {
public:
  CUDA_CALLABLE_MEMBER HiRaceDataWrap3D(T (*data)[yDim][zDim]) : _data(data),
                                                                 _scope(Scope::Tid),
                                                                 _dim(3),
                                                                 _offset_z(0),
                                                                 _yWrapper(data[0]) {}
  
  CUDA_CALLABLE_MEMBER HiRaceDataWrap3D(T (*data)[yDim][zDim],
                                        hr_shadowt (*metadata)[yDim][zDim],
                                        Scope scope,
                                        unsigned *bcount,
                                        unsigned *wcount,
                                        unsigned *swidx,
                                        int dim) : 
                                          _data(data),
                                          _metadata(metadata),
                                          _bcount(bcount),
                                          _wcount(wcount),
                                          _swidx(swidx),
                                          _scope(scope),
                                          _dim(dim),
                                          _offset_z(0),
                                          _yWrapper(data[0]) {}
  
  CUDA_CALLABLE_MEMBER HiRaceDataWrap3D(HiRaceDataWrap3D<T, xDim, yDim, zDim> &other) = default;

  CUDA_CALLABLE_MEMBER void setMembers(T (*data)[yDim][zDim],
                                       hr_shadowt (*metadata)[yDim][zDim],
                                       Scope scope,
                                       unsigned *bcount,
                                       unsigned *wcount,
                                       unsigned *swidx,
                                       int dim) {
    _data = data;
    _metadata = metadata;
    _bcount = bcount;
    _wcount = wcount;
    _swidx = swidx;
    _scope = scope;
    _dim = dim;
    _offset_z = 0;
  }
 
  CUDA_CALLABLE_MEMBER operator T*() { return _data; }
  
  CUDA_CALLABLE_MEMBER HiRaceDataWrap2D<T, xDim, yDim>& operator[](int idx) {
    _offset_z = idx;
    if (_scope == Scope::Tid) {
      _yWrapper.setMembers(_data[idx],
                           nullptr,
                           _scope,
                           _bcount,
                           _wcount,
                           _swidx,
                           _dim,
                           _offset_z);
    }
    else {
      _yWrapper.setMembers(_data[idx],
                           _metadata[idx],
                           _scope,
                           _bcount,
                           _wcount,
                           _swidx,
                           _dim,
                           _offset_z);
    }
    
    return _yWrapper.registerCallsite(_lineNo, _fileName);
  }
  
  CUDA_CALLABLE_MEMBER HiRaceDataWrap3D& registerCallsite(int lineNo, const char *fileName) {
      _lineNo = lineNo;
      _fileName = fileName;
      return *this;
  }

private:
  T (*_data)[yDim][zDim];
  hr_shadowt (*_metadata)[yDim][zDim];
  
  HiRaceDataWrap2D<T, xDim, yDim> _yWrapper;
  
  unsigned *_bcount;
  unsigned *_wcount;
  unsigned *_swidx;
  Scope _scope;
  int _dim;
  int _offset_z;
  int _lineNo;
  const char *_fileName;
}; // HiRaceDataWrap3D<T>

#endif // HIRACEDATAWRAP_H
