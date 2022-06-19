from peewee import IntegerField, TextField, AutoField, Model, SqliteDatabase, CharField

db = SqliteDatabase('chant_oracle.db')


class BaseModel(Model):
    class Meta:
        database = db


class OracleBoneItems(BaseModel):
    """
    甲片条目表，每条数据代表甲片中的一个条目，(著录号-条号) 唯一对应一条数据
    """
    # 自增 id 域，主键
    id = AutoField()
    # 著录号（甲片的唯一标识），最大长度 511 字符, 原 book_name
    published_collection_number = CharField(null=False, max_length=511, column_name='published_collection_number')
    # 汉达条号（正整数，甲片下某个甲骨文句子的序号），原 row_order
    chant_notation_number = IntegerField(null=False, column_name='chant_notation_number')
    # 汉达释文，甲骨文句子（繁体汉字），带 font 标签，原 modern_text
    chant_transcription_text = TextField(null=False, column_name='chant_transcription_text')
    # 汉达文库类别，最大长度 7，原 category
    chant_calligraphy = CharField(null=False, max_length=7, column_name='chant_calligraphy')
    # 源数据在汉达文库中的 url 后缀，最大长度 511 字符，原 url
    chant_url = CharField(null=False, max_length=511, column_name='chant_url')
    # 包含单字的列表，以 '\t' 分隔的字符串，每个元素都是字形表 CharShape.id
    characters = TextField(null=False, column_name='characters')
    # 甲片图的路径，最大长度 511 字符，原 l_bone_img
    chant_processed_rubbing = CharField(null=False, max_length=511, column_name='chant_processed_rubbing')
    # 汉字排布图的路径，最大长度 511 字符，原 r_bone_img
    chant_mapped_character_image = CharField(null=False, max_length=511, column_name='chant_mapped_character_image')
    # 其他信息
    meta = TextField(null=False, default='{}', column_name='meta')

    class Meta:
        # 著录号-条号 联合唯一索引
        indexes = (
            (('published_collection_number', 'chant_notation_number'), True),
        )


class Character(BaseModel):
    """
    单字表，一个单字包含多个字形，(编码-字体) 不一定唯一对应一个单字！同一个汉字可能对应多个 Character（编码不同）!
    """
    # 自增 id 域，主键
    id = AutoField()
    # 摹本中的编号，索引字段，为 -1 表示只在汉达而不在摹本中的字，原 char_index
    wzb_character_number = IntegerField(null=False, index=True, column_name='wzb_character_number')
    # 对应的现代汉字，utf-8 编码，原 char_byte
    modern_character = CharField(null=False, max_length=7, column_name='modern_character')
    # 汉达字体标签 name，原 font
    chant_font_label = CharField(null=False, max_length=7, column_name='chant_font_label')
    # 部首编号，未指定时为 -1
    wzb_radical = IntegerField(null=False, default=-1, column_name='wzb_radical')
    # 其他信息
    meta = TextField(null=False, default='{}', column_name='meta')

    class Meta:
        # 编码-字体 联合唯一索引
        indexes = (
            (('modern_character', 'chant_font_label'), False),
        )


class CharFace(BaseModel):
    """
    字形表，对应汉达文库/李老师摹本中的每一个单字字形
    """
    # 自增 id 域，主键
    id = AutoField()
    # 根据 (著录号 - 无字体汉字) 进行匹配
    # 0-只在汉达中，不在摹本中；1-只在摹本中，不在汉达中；2-同时存在，数据可以对上，一一对应；3-同时存在，数据可对上，多于1条
    match_case = IntegerField(null=False, index=True, column_name='match_case')
    # 属于哪一个单字 Character.id，索引字段，原 char_belong
    liding_character = IntegerField(null=False, index=True, column_name='liding_character')
    # 在汉达文库甲片图中的坐标信息，可能为空（match_case == 1），原 coords
    chant_coordinates = CharField(null=False, max_length=63, column_name='chant_coordinates')
    # 原形，即汉达文库中带背景的噪声图片路径，最大长度 511 字符，可能为空（match_case == 1），原 noise_image
    chant_authentic_face = CharField(null=False, max_length=511, column_name='chant_authentic_face')
    # 摹写字形图片路径，最大长度 511 字符，可能为空（match_case == 0），原 shape_image
    wzb_handcopy_face = CharField(null=False, max_length=511, column_name='wzb_handcopy_face')
    # 所属的著录号，最大长度 511 字符，match_case == 0/2-取汉达著录号表示，1-取摹本著录号表示，原 book_name
    published_collection_number = CharField(null=False, max_length=511, column_name='published_collection_number')
    # 李老师摹本类别码，最大长度 7，可能为空（match_case == 0），missing 表示找不到有效 ocr 编码，原 category
    wzb_calligraphy = CharField(null=False, max_length=7, column_name='wzb_calligraphy')
    # 页码号，可能为 -1（match_case == 0），原 page_number
    wzb_page_number = IntegerField(null=False, column_name='wzb_page_number')
    # 第几行，可能为 -1（match_case == 0）
    wzb_row_number = IntegerField(null=False, column_name='wzb_row_number')
    # 第几列，可能为 -1（match_case == 0）
    wzb_col_number = IntegerField(null=False, column_name='wzb_col_number')
    # 其他信息
    meta = TextField(null=False, default='{}', column_name='meta')


def init_db():
    db.connection()