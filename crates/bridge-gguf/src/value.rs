use crate::GgufError;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum GgufValueType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl TryFrom<u32> for GgufValueType {
    type Error = GgufError;

    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::U8),
            1 => Ok(Self::I8),
            2 => Ok(Self::U16),
            3 => Ok(Self::I16),
            4 => Ok(Self::U32),
            5 => Ok(Self::I32),
            6 => Ok(Self::F32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::U64),
            11 => Ok(Self::I64),
            12 => Ok(Self::F64),
            other => Err(GgufError::UnknownValueType(other)),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(GgufArray),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    pub const fn value_type(&self) -> GgufValueType {
        match self {
            Self::U8(_) => GgufValueType::U8,
            Self::I8(_) => GgufValueType::I8,
            Self::U16(_) => GgufValueType::U16,
            Self::I16(_) => GgufValueType::I16,
            Self::U32(_) => GgufValueType::U32,
            Self::I32(_) => GgufValueType::I32,
            Self::F32(_) => GgufValueType::F32,
            Self::Bool(_) => GgufValueType::Bool,
            Self::String(_) => GgufValueType::String,
            Self::Array(_) => GgufValueType::Array,
            Self::U64(_) => GgufValueType::U64,
            Self::I64(_) => GgufValueType::I64,
            Self::F64(_) => GgufValueType::F64,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GgufArray {
    pub element_type: GgufValueType,
    pub values: Vec<GgufValue>,
}
